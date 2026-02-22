"""
src/chunker.py
──────────────
Split transcript text into token-bounded chunks with metadata.
Uses tiktoken (cl100k_base) for exact token counting.
"""

import os
import tiktoken
from dataclasses import dataclass, asdict

ENCODING_NAME = "cl100k_base"  # matches text-embedding-3-small & GPT-4o


@dataclass
class Chunk:
    video_id: str
    video_title: str
    video_url: str
    chunk_index: int
    chunk_text: str
    token_count: int
    start_time_sec: float  # approximate, from closest transcript segment


def _get_encoder() -> tiktoken.Encoding:
    return tiktoken.get_encoding(ENCODING_NAME)


def count_tokens(text: str) -> int:
    """Return the number of tokens in *text* using cl100k_base."""
    enc = _get_encoder()
    return len(enc.encode(text))


def chunk_transcript(
    segments: list[dict],
    video_id: str,
    video_title: str,
    video_url: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[Chunk]:
    """
    Split transcript segments into overlapping token-bounded chunks.

    Each segment is a dict: {"text": str, "start": float, "duration": float}

    Strategy:
      - Accumulate full text word-by-word from segments.
      - When a chunk reaches chunk_size tokens, store it and start the next
        chunk with the last *overlap* tokens of the previous one.
      - Record the start_time_sec of the first segment that contributed to
        each chunk.

    Returns a list of Chunk objects.
    """
    chunk_size = chunk_size or int(os.getenv("CHUNK_SIZE_TOKENS", "500"))
    overlap = overlap or int(os.getenv("CHUNK_OVERLAP_TOKENS", "50"))

    enc = _get_encoder()

    chunks: list[Chunk] = []
    current_tokens: list[int] = []   # token IDs of the current chunk
    current_start_time: float = 0.0
    chunk_index = 0

    for seg_idx, seg in enumerate(segments):
        seg_text = seg["text"].strip().replace("\n", " ")
        seg_tokens = enc.encode(seg_text + " ")

        # If this is the very first segment contributing to this chunk,
        # record its start time.
        if not current_tokens:
            current_start_time = seg.get("start", 0.0)

        current_tokens.extend(seg_tokens)

        # When we exceed chunk_size, flush the chunk
        while len(current_tokens) >= chunk_size:
            chunk_text = enc.decode(current_tokens[:chunk_size])
            chunks.append(
                Chunk(
                    video_id=video_id,
                    video_title=video_title,
                    video_url=video_url,
                    chunk_index=chunk_index,
                    chunk_text=chunk_text,
                    token_count=chunk_size,
                    start_time_sec=current_start_time,
                )
            )
            chunk_index += 1

            # Keep the overlap tokens and reset start time
            current_tokens = current_tokens[chunk_size - overlap:]
            # Approximate: next chunk starts at current segment start
            current_start_time = seg.get("start", 0.0)

    # Flush any remaining tokens as the final (possibly shorter) chunk
    if current_tokens:
        remaining_text = enc.decode(current_tokens)
        chunks.append(
            Chunk(
                video_id=video_id,
                video_title=video_title,
                video_url=video_url,
                chunk_index=chunk_index,
                chunk_text=remaining_text,
                token_count=len(current_tokens),
                start_time_sec=current_start_time,
            )
        )

    return chunks


def chunks_to_dicts(chunks: list[Chunk]) -> list[dict]:
    """Convert list of Chunk dataclasses to plain dicts for serialization."""
    return [asdict(c) for c in chunks]
