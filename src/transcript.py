"""
src/transcript.py
─────────────────
Extract transcripts from YouTube videos via youtube-transcript-api v1.x.
Falls back gracefully if no captions are available.

v1.x API changes:
  - YouTubeTranscriptApi must be INSTANTIATED: ytt = YouTubeTranscriptApi()
  - list_transcripts() → list()
  - fetch() returns FetchedTranscript; call .to_raw_data() for plain dicts
  - Simple fetch: ytt.fetch(video_id, languages=[...])
"""

import re
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
)


def get_video_id(url: str) -> str:
    """
    Extract the 11-character YouTube video ID from any common URL format.

    Supported formats:
      - https://www.youtube.com/watch?v=VIDEO_ID
      - https://youtu.be/VIDEO_ID
      - https://www.youtube.com/embed/VIDEO_ID
      - https://www.youtube.com/shorts/VIDEO_ID
    """
    patterns = [
        r"(?:v=)([A-Za-z0-9_-]{11})",
        r"youtu\.be/([A-Za-z0-9_-]{11})",
        r"embed/([A-Za-z0-9_-]{11})",
        r"shorts/([A-Za-z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError(
        f"Could not extract a valid YouTube video ID from URL: {url}\n"
        "Please make sure you're using a standard YouTube link."
    )


def fetch_transcript(video_id: str, languages: list[str] | None = None) -> list[dict]:
    """
    Fetch the transcript for a YouTube video using youtube-transcript-api v1.x.

    Returns a list of plain dicts:
        [{"text": str, "start": float, "duration": float}, ...]
    """
    langs = languages or ["en", "en-US", "en-GB"]
    ytt = YouTubeTranscriptApi()

    try:
        # Simple path: fetch directly with language preference list
        fetched = ytt.fetch(video_id, languages=langs)
        return fetched.to_raw_data()

    except NoTranscriptFound:
        pass  # fall through to try listing all available transcripts
    except TranscriptsDisabled as exc:
        raise TranscriptsDisabled(video_id) from exc
    except Exception as exc:
        raise ValueError(f"Failed to fetch transcript for {video_id}: {exc}") from exc

    # Fallback: list all transcripts, pick the first one and translate to English
    try:
        transcript_list = ytt.list(video_id)
        # find_transcript tries manual first, then auto-generated
        transcript = transcript_list.find_transcript(
            [t.language_code for t in transcript_list]
        )
        if transcript.is_translatable:
            fetched = transcript.translate("en").fetch()
        else:
            fetched = transcript.fetch()
        return fetched.to_raw_data()

    except Exception as exc:
        raise ValueError(
            f"No usable transcript found for video {video_id}. "
            f"Make sure captions are enabled on the video. Details: {exc}"
        ) from exc


def segments_to_full_text(segments: list[dict]) -> str:
    """Concatenate all segment texts into a single string."""
    return " ".join(seg["text"].strip() for seg in segments)


def get_video_metadata(video_id: str) -> dict:
    """Return lightweight metadata dict (URL only; title set by user in UI)."""
    return {
        "video_id": video_id,
        "video_url": f"https://www.youtube.com/watch?v={video_id}",
    }

