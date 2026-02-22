"""
src/qa_engine.py
────────────────
Full RAG Q&A pipeline:
  1. Embed the user's query
  2. Retrieve top-k similar chunks from Supabase
  3. Build a prompt with the retrieved context
  4. Call the LLM and return the answer

If the retrieved context is empty, or the LLM determines the question is
unrelated to the video, returns the standard "irrelevant" message.
"""

import os
import logging
from openai import OpenAI
from src.embedder import embed_query
from src.vector_store import similarity_search

logger = logging.getLogger(__name__)

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
IRRELEVANT_RESPONSE = (
    "The question is irrelevant to this video. "
    "Please ask something related to the video content."
)

SYSTEM_PROMPT = """\
You are an expert assistant that answers questions strictly based on the
transcript excerpt(s) of a YouTube video provided to you.

Rules:
1. Base your answer ONLY on the provided context. Do not use outside knowledge.
2. If the context does not contain enough information to answer the question,
   reply with exactly: "The question is irrelevant to this video."
3. Be concise, accurate, and cite the approximate timestamp when possible.
4. Never fabricate information.
"""


def _build_user_prompt(query: str, chunks: list[dict]) -> str:
    context_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        start = chunk.get("start_time_sec", 0)
        minutes = int(start // 60)
        seconds = int(start % 60)
        timestamp = f"~{minutes}m {seconds}s" if start else "N/A"
        context_blocks.append(
            f"[Excerpt {i} | Timestamp: {timestamp}]\n{chunk['chunk_text']}"
        )
    context = "\n\n---\n\n".join(context_blocks)

    return f"""\
Video context (transcript excerpts):

{context}

---

User question: {query}

Answer based only on the context above. If you cannot answer from the context, \
reply with exactly: "The question is irrelevant to this video."
"""


def answer_query(
    query: str,
    video_id: str,
    video_title: str = "",
    top_k: int | None = None,
) -> dict:
    """
    Full RAG pipeline for a single user query.

    Returns a dict:
    {
        "answer":        str,
        "chunks_used":   int,
        "chunks":        list[dict],   # retrieved chunks with similarity scores
        "is_irrelevant": bool,
    }
    """
    top_k = top_k or int(os.getenv("TOP_K_RETRIEVAL", "5"))

    # 1. Embed the query
    logger.info("Embedding query: %s", query)
    query_embedding = embed_query(query)

    # 2. Retrieve similar chunks
    logger.info("Searching for top-%d chunks in video %s", top_k, video_id)
    chunks = similarity_search(query_embedding, video_id, top_k=top_k)

    if not chunks:
        return {
            "answer": IRRELEVANT_RESPONSE,
            "chunks_used": 0,
            "chunks": [],
            "is_irrelevant": True,
        }

    # 3. Build prompt and call LLM
    user_prompt = _build_user_prompt(query, chunks)
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = os.getenv("LLM_MODEL", LLM_MODEL)

    logger.info("Calling LLM (%s) for answer…", model)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=800,
    )

    answer = response.choices[0].message.content.strip()
    is_irrelevant = "irrelevant" in answer.lower()

    return {
        "answer": answer,
        "chunks_used": len(chunks),
        "chunks": chunks,
        "is_irrelevant": is_irrelevant,
    }
