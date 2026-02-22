"""
src/vector_store.py
───────────────────
Supabase + pgvector integration.

Operations:
  - upsert_chunks  : store embeddings + metadata into video_chunks table
  - similarity_search : retrieve top-k similar chunks via RPC
  - video_is_indexed  : check if a video ID is already in the DB
  - delete_video      : remove all chunks for a given video ID
"""

import os
import logging
from supabase import create_client, Client

logger = logging.getLogger(__name__)

TABLE = "video_chunks"
MATCH_FN = "match_video_chunks"


def _get_client() -> Client:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise EnvironmentError(
            "SUPABASE_URL and SUPABASE_KEY must be set in your .env file."
        )
    return create_client(url, key)


def upsert_chunks(
    chunks: list[dict],
    embeddings: list[list[float]],
) -> int:
    """
    Insert chunks with their embeddings into the video_chunks table.

    Parameters
    ----------
    chunks      : list of dicts from chunker.chunks_to_dicts()
    embeddings  : parallel list of embedding vectors

    Returns
    -------
    Number of rows inserted.
    """
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must have the same length.")

    client = _get_client()
    rows = []
    for chunk, emb in zip(chunks, embeddings):
        rows.append(
            {
                "video_id": chunk["video_id"],
                "video_title": chunk["video_title"],
                "video_url": chunk["video_url"],
                "chunk_index": chunk["chunk_index"],
                "chunk_text": chunk["chunk_text"],
                "token_count": chunk["token_count"],
                "start_time_sec": chunk["start_time_sec"],
                "embedding": emb,
            }
        )

    # Batch insert (Supabase supports up to ~1000 rows per call)
    BATCH = 200
    total_inserted = 0
    for i in range(0, len(rows), BATCH):
        batch = rows[i : i + BATCH]
        result = client.table(TABLE).insert(batch).execute()
        total_inserted += len(result.data)
        logger.info("Inserted rows %d–%d", i + 1, i + len(batch))

    return total_inserted


def similarity_search(
    query_embedding: list[float],
    video_id: str,
    top_k: int | None = None,
) -> list[dict]:
    """
    Find the top-k most similar chunks for a given video via the
    match_video_chunks SQL function (cosine similarity).

    Returns a list of dicts with keys:
        id, video_id, video_title, chunk_index, chunk_text,
        token_count, start_time_sec, similarity
    """
    top_k = top_k or int(os.getenv("TOP_K_RETRIEVAL", "5"))
    client = _get_client()

    response = client.rpc(
        MATCH_FN,
        {
            "query_embedding": query_embedding,
            "match_video_id": video_id,
            "match_count": top_k,
        },
    ).execute()

    return response.data or []


def video_is_indexed(video_id: str) -> bool:
    """Return True if any chunks for video_id already exist in the table."""
    client = _get_client()
    response = (
        client.table(TABLE)
        .select("id", count="exact")
        .eq("video_id", video_id)
        .limit(1)
        .execute()
    )
    return (response.count or 0) > 0


def delete_video(video_id: str) -> int:
    """Delete all chunks for a given video. Returns number of deleted rows."""
    client = _get_client()
    response = client.table(TABLE).delete().eq("video_id", video_id).execute()
    deleted = len(response.data) if response.data else 0
    logger.info("Deleted %d chunks for video %s", deleted, video_id)
    return deleted

##