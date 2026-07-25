"""
src/chunker.py
──────────────
Sentence-aware, token-bounded chunking of YouTube transcripts.

Strategy:
  1. Collect all transcript segment text into one string.
  2. Split into sentences using a simple regex (no extra dependencies).
  3. Greedily accumulate whole sentences into a chunk until adding the
     next sentence would exceed `chunk_size` tokens — then close the
     chunk and start a new one.
  4. Overlap: seed each new chunk with the last `overlap_sentences`
     sentences from the previous chunk, so context at boundaries is
     preserved.
  5. Track exact token count per chunk using tiktoken (cl100k_base).

Why sentence-aware?
  Pure token-slicing can cut "...attention mecha|nisms..." mid-word.
  Sentence-aware chunking ensures every chunk starts and ends at a
  natural sentence boundary → better retrieval quality.
"""

import os
import re
import tiktoken
from dataclasses import dataclass, asdict

ENCODING_NAME = "cl100k_base"  # matches text-embedding-3-small & GPT-4o
OVERLAP_SENTENCES = 2          # sentences seeded into start of next chunk
DEFAULT_CHUNK_SIZE = 400       # safe default — keeps chunks well under 512 tokens


@dataclass
class Chunk:
    video_id: str
    video_title: str
    video_url: str
    chunk_index: int
    chunk_text: str
    token_count: int
    start_time_sec: float   # approximate timestamp of the first sentence in chunk


def _get_encoder() -> tiktoken.Encoding:
    return tiktoken.get_encoding(ENCODING_NAME)


def count_tokens(text: str) -> int:
    """Return the exact token count for *text* using cl100k_base."""
    return len(_get_encoder().encode(text))


def _split_sentences(text: str) -> list[str]:
    """
    Split text into sentences using punctuation boundaries.
    Handles common abbreviations well enough for transcript text.
    Returns a list of non-empty sentence strings.
    """
    # Split on . ! ? followed by whitespace or end-of-string
    raw = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in raw if s.strip()]


def chunk_transcript(
    segments: list[dict],
    video_id: str,
    video_title: str,
    video_url: str,
    chunk_size: int | None = None,
    overlap: int | None = None,         # kept for API compat; overlap is sentence-based
    overlap_sentences: int = OVERLAP_SENTENCES,
) -> list["Chunk"]:
    """
    Split transcript segments into sentence-aware, token-bounded chunks.

    Each segment: {"text": str, "start": float, "duration": float}

    Parameters
    ----------
    chunk_size        : max tokens per chunk (default: CHUNK_SIZE_TOKENS env var or 500)
    overlap_sentences : how many sentences from the end of the previous chunk
                        to carry into the start of the next chunk (default: 2)

    Returns a list of Chunk objects, each containing full sentences only.
    """
    max_tokens = chunk_size or int(os.getenv("CHUNK_SIZE_TOKENS", str(DEFAULT_CHUNK_SIZE)))
    enc = _get_encoder()

    # ── Build a flat list of (sentence, approx_start_time_sec) ──────────────
    sentence_pairs: list[tuple[str, float]] = []
    for seg in segments:
        seg_text = seg["text"].strip().replace("\n", " ")
        seg_start = seg.get("start", 0.0)
        for sent in _split_sentences(seg_text):
            sentence_pairs.append((sent, seg_start))

    if not sentence_pairs:
        return []

    # ── Greedily accumulate sentences into chunks ────────────────────────────
    # Sentence-boundary guarantee:
    #   - A chunk is ONLY closed after a complete sentence. The inner loop
    #     breaks BEFORE appending the offending sentence, not after, so the
    #     last sentence in every chunk is always complete.
    #   - The NEXT chunk is seeded with the last `overlap_sentences` complete
    #     sentences from the chunk just closed, so it also starts on a full
    #     sentence boundary.
    #   → Every chunk starts AND ends at a full sentence boundary.
    chunks: list[Chunk] = []
    chunk_index = 0
    seed_sents: list[str] = []   # overlap sentences carried forward

    i = 0  # pointer into sentence_pairs

    while i < len(sentence_pairs):
        # Seed this chunk with overlap sentences from the previous chunk
        current_sents: list[str] = list(seed_sents)
        current_start_time: float = sentence_pairs[i][1]

        # Accumulate whole sentences until the NEXT sentence would exceed limit
        while i < len(sentence_pairs):
            sent, _ = sentence_pairs[i]
            candidate = " ".join(current_sents + [sent])
            if len(enc.encode(candidate)) > max_tokens and current_sents:
                # Next sentence would exceed max_tokens — close chunk HERE.
                # The current last sentence is already complete.
                break
            current_sents.append(sent)
            i += 1

        if not current_sents:
            # Edge case: single sentence longer than max_tokens.
            # Accept it alone to avoid infinite loop.
            current_sents.append(sentence_pairs[i][0])
            i += 1

        chunk_text  = " ".join(current_sents)
        token_count = len(enc.encode(chunk_text))

        chunks.append(
            Chunk(
                video_id=video_id,
                video_title=video_title,
                video_url=video_url,
                chunk_index=chunk_index,
                chunk_text=chunk_text,
                token_count=token_count,
                start_time_sec=current_start_time,
            )
        )
        chunk_index += 1

        # Build overlap seed for next chunk:
        # Take the last `overlap_sentences` of the NEW sentences added this
        # round (excluding sentences that were already seed from last chunk).
        new_sents = current_sents[len(seed_sents):]
        seed_sents = new_sents[-overlap_sentences:] if overlap_sentences > 0 else []

    return chunks


def chunks_to_dicts(chunks: list[Chunk]) -> list[dict]:
    """Convert list of Chunk dataclasses to plain dicts."""
    return [asdict(c) for c in chunks]
