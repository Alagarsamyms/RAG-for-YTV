"""
src/embedder.py
───────────────
Generate embeddings via OpenAI text-embedding-3-small.
Includes batching and retry logic with exponential back-off.
"""

import os
import time
import logging
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = 1536   # fixed for text-embedding-3-small
BATCH_SIZE = 100        # max texts per API call (safe limit)


def _get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set in your environment.")
    return OpenAI(api_key=api_key)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def _embed_batch(client: OpenAI, texts: list[str], model: str) -> list[list[float]]:
    """Call OpenAI embeddings API for a single batch of texts."""
    response = client.embeddings.create(
        input=texts,
        model=model,
    )
    return [item.embedding for item in response.data]


def embed_texts(
    texts: list[str],
    model: str | None = None,
) -> list[list[float]]:
    """
    Generate embeddings for a list of texts.

    Automatically batches large lists and retries on transient errors.

    Returns:
        List of embedding vectors (each a list of floats), same order as input.
    """
    model = model or EMBEDDING_MODEL
    client = _get_client()

    all_embeddings: list[list[float]] = []
    total = len(texts)

    for batch_start in range(0, total, BATCH_SIZE):
        batch = texts[batch_start: batch_start + BATCH_SIZE]
        logger.info(
            "Embedding batch %d–%d of %d texts…",
            batch_start + 1,
            min(batch_start + BATCH_SIZE, total),
            total,
        )
        batch_embeddings = _embed_batch(client, batch, model)
        all_embeddings.extend(batch_embeddings)

        # Small courtesy delay between large batches to avoid rate limits
        if batch_start + BATCH_SIZE < total:
            time.sleep(0.3)

    return all_embeddings


def embed_query(query: str, model: str | None = None) -> list[float]:
    """Generate a single embedding for a search query."""
    return embed_texts([query], model=model)[0]
