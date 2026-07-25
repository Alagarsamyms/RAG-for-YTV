# RAG for YTV (YouTube Video) - Project Overview

## 1. Problem
Long-form video content on YouTube contains vast amounts of valuable information, but extracting specific knowledge or answering precise questions requires users to watch the entire video or scrub through timelines manually. Traditional keyword searches on transcripts often fail to capture context or semantic meaning, making it time-consuming and inefficient for users to find the exact information they need.

## 2. Product Vision
To build a seamless and intelligent interactive application that allows users to "chat" with any YouTube video. The vision is to abstract away the complexities of audio transcription, vector mathematics, and language model orchestration, providing a simple interface where users can ask natural language questions and receive accurate, context-grounded answers with citations directly from the video's content.

## 3. Your Role
In this project, I wore multiple hats spanning the entire software development lifecycle:
* **Product Manager:** Defined the core user journey (from pasting a URL to asking questions), prioritized features for the MVP (Streamlit UI, robust error handling for missing transcripts), and ensured the final product solved the core user problem effectively.
* **Business Analyst (BA):** Mapped out the logical workflows for data ingestion. Designed the rules around sentence-aware sliding window chunking to ensure data coherence, defined token limits, and analyzed the trade-offs between chunk sizes and retrieval accuracy.
* **AI Engineer:** Architected and implemented the end-to-end Retrieval-Augmented Generation (RAG) pipeline. Integrated OpenAI models for embeddings and answer generation, managed vector storage in Supabase using pgvector, and wrote the backend logic for transcription extraction and semantic search.

## 4. Key Features
* **Automated Transcript Extraction:** Instantly fetches timed caption segments from any YouTube video URL using `youtube-transcript-api`.
* **Intelligent Sentence-Aware Chunking:** Uses `tiktoken` to split text into token-bounded, complete-sentence chunks with overlap. This prevents mid-sentence cuts and dramatically improves the LLM's comprehension and retrieval quality.
* **Vector Database Integration:** Stores video metadata and 1536-dimensional embeddings in a Supabase PostgreSQL database using the `pgvector` extension.
* **Context-Grounded Q&A Engine:** Uses cosine similarity search to retrieve the most relevant transcript segments, feeding them to GPT-4o-mini to generate accurate answers that are strictly bound to the video context (preventing hallucinations).
* **Interactive Web Interface:** A user-friendly Streamlit web application that visualizes the process, allows configuration of RAG parameters (chunk size, overlap, top-K retrieval), and displays retrieved context and timestamps alongside the answer.

## 5. Business Impact / Learning
* **Outcome:** Delivered a fully functional, production-ready AI tool that drastically reduces the time required to consume and query video content.
* **Key Insights & Learning:** 
  * Learned that raw token-slicing degrades RAG performance; implementing sentence-aware sliding window chunking was a crucial breakthrough for answer quality.
  * Gained hands-on expertise with vector databases (Supabase/pgvector) and writing custom SQL functions (`ivfflat` indexing, `supabase.rpc()` for cosine similarity).
  * Mastered the orchestration of multiple API layers (YouTube, OpenAI, Supabase) while handling edge cases like missing captions and rate limits gracefully.

## 6. Technical Overview (End-to-End)
The system architecture follows a classic but highly optimized RAG (Retrieval-Augmented Generation) pattern:

1. **Ingestion & Processing (`src/transcript.py` & `src/chunker.py`):**
   * The user provides a YouTube URL in the Streamlit UI (`app.py`).
   * The system extracts the video ID and fetches the transcript.
   * The transcript segments are flattened and processed by the chunker. It greedily accumulates whole sentences until reaching the max token limit (default: 500 tokens), carrying over a set number of sentences (default: 2) as an overlap into the next chunk.

2. **Embedding & Storage (`src/embedder.py` & `src/vector_store.py`):**
   * The chunks are sent to OpenAI's `text-embedding-3-small` API in batches to generate 1536-dimensional vector embeddings.
   * These chunks, along with their metadata and embeddings, are upserted into a Supabase PostgreSQL database via a batch API call. The database utilizes an `ivfflat` index on the `pgvector` extension for rapid retrieval.

3. **Retrieval & Generation (`src/qa_engine.py`):**
   * When a user asks a question, the query is immediately embedded using the same OpenAI embedding model.
   * A remote procedure call (RPC) is made to Supabase (`match_video_chunks`), performing a cosine similarity search between the query vector and the stored chunk vectors for that specific video.
   * The top-K most relevant chunks are retrieved and formatted into a structured prompt alongside the user's question.
   * GPT-4o-mini (`LLM_MODEL`) is invoked with a strict system prompt instructing it to answer *only* based on the provided excerpts. If the context does not contain the answer, it returns a standard "irrelevant" response.

4. **Presentation (`app.py`):**
   * The final answer is streamed back to the user via the Streamlit interface, complete with expandable drop-downs showing exactly which transcript excerpts and timestamps were used to generate the response.
