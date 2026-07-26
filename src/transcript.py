"""
src/transcript.py
─────────────────
Extract transcripts from YouTube videos.

Fetch strategy (tried in order, automatic):
  1. youtube-transcript-api v1.x   — fast, works perfectly on local dev.
  2. yt-dlp (android player client) — fallback for cloud deployments where
       YouTube blocks direct browser-style requests from datacenter IPs.
       The android/TV innertube endpoint is subject to much lighter IP
       filtering and usually succeeds from Streamlit Cloud, AWS, GCP, etc.

Proxy support (optional, for extra reliability):
  - Set WEBSHARE_PROXY_USERNAME + WEBSHARE_PROXY_PASSWORD   → Webshare
  - Set HTTP_PROXY / HTTPS_PROXY                            → any provider
  build_proxy_config() reads these and returns the right ProxyConfig object
  (or None for local dev — zero impact).

No sign-up or external account is required for the yt-dlp fallback path.
"""

import os
import re
import logging

logger = logging.getLogger(__name__)


# ── Proxy helper ──────────────────────────────────────────────────────────────

def build_proxy_config():
    """
    Build a proxy_config object for YouTubeTranscriptApi from environment vars.

    Priority:
      1. Webshare residential proxy  → WEBSHARE_PROXY_USERNAME + WEBSHARE_PROXY_PASSWORD
      2. Generic HTTP/HTTPS proxy    → HTTP_PROXY / HTTPS_PROXY (standard env vars)
      3. No proxy (local dev)        → returns None

    Returns a proxy_config object accepted by YouTubeTranscriptApi(), or None.
    """
    # ── Option 1: Webshare residential proxy ─────────────────────────────────
    ws_user = os.getenv("WEBSHARE_PROXY_USERNAME", "").strip()
    ws_pass = os.getenv("WEBSHARE_PROXY_PASSWORD", "").strip()
    if ws_user and ws_pass:
        try:
            from youtube_transcript_api.proxies import WebshareProxyConfig
            return WebshareProxyConfig(
                proxy_username=ws_user,
                proxy_password=ws_pass,
            )
        except ImportError:
            pass  # fall through to generic proxy

    # ── Option 2: Generic HTTP/HTTPS proxy (any provider) ────────────────────
    http_proxy = os.getenv("HTTP_PROXY", "").strip() or os.getenv("http_proxy", "").strip()
    https_proxy = os.getenv("HTTPS_PROXY", "").strip() or os.getenv("https_proxy", "").strip()
    if http_proxy or https_proxy:
        try:
            from youtube_transcript_api.proxies import GenericProxyConfig
            return GenericProxyConfig(
                http_url=http_proxy or https_proxy,
                https_url=https_proxy or http_proxy,
            )
        except ImportError:
            pass

    # ── Option 3: No proxy (local dev — works fine) ───────────────────────────
    return None


# ── URL parser ────────────────────────────────────────────────────────────────

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


# ── json3 subtitle parser (used by yt-dlp path) ───────────────────────────────

def _parse_json3(data: dict) -> list[dict]:
    """
    Parse YouTube's json3 timed-text format into the same flat list of dicts
    that youtube-transcript-api returns:
        [{"text": str, "start": float, "duration": float}, ...]
    """
    segments = []
    for event in data.get("events", []):
        segs = event.get("segs")
        if not segs:
            continue
        text = "".join(s.get("utf8", "") for s in segs).strip().replace("\n", " ")
        if not text:
            continue
        segments.append({
            "text": text,
            "start": event.get("tStartMs", 0) / 1000.0,
            "duration": event.get("dDurationMs", 0) / 1000.0,
        })
    return segments


# ── Strategy 1: youtube-transcript-api ───────────────────────────────────────

def _fetch_via_ytt(
    video_id: str,
    languages: list[str],
    proxy_config,
) -> list[dict]:
    """
    Fetch via youtube-transcript-api v1.x.
    Raises on any failure so the caller can fall through to the next strategy.
    """
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound

    kwargs = {"proxy_config": proxy_config} if proxy_config is not None else {}
    ytt = YouTubeTranscriptApi(**kwargs)

    try:
        fetched = ytt.fetch(video_id, languages=languages)
        return fetched.to_raw_data()
    except NoTranscriptFound:
        pass  # try listing all available transcripts below
    # Let TranscriptsDisabled and RequestBlocked/IpBlocked propagate up

    # Fallback within ytt: list all transcripts, take the first, translate if possible
    transcript_list = ytt.list(video_id)
    transcript = transcript_list.find_transcript(
        [t.language_code for t in transcript_list]
    )
    if transcript.is_translatable:
        fetched = transcript.translate("en").fetch()
    else:
        fetched = transcript.fetch()
    return fetched.to_raw_data()


# ── Strategy 2: yt-dlp (android / web innertube client) ──────────────────────

def _fetch_via_ytdlp(video_id: str, languages: list[str]) -> list[dict]:
    """
    Fetch transcript via yt-dlp using YouTube's android + web player clients.

    Why this works on cloud IPs:
      - yt-dlp queries YouTube's internal innertube API with client credentials
        that mimic the YouTube Android app.
      - The android/TV innertube endpoint (youtubei/v1/player) uses much lighter
        IP-based rate limiting than the browser-facing watch page, so it
        succeeds from AWS / GCP / Streamlit Cloud IPs where the browser path
        is blocked.
      - The subtitle file URL obtained from the player response is then fetched
        separately with a plain HTTP GET — no special auth required.

    No proxy, no account, no API key needed.
    """
    try:
        import yt_dlp
        import requests as req
    except ImportError as e:
        raise ImportError(
            "yt-dlp is required for the cloud fallback path. "
            "Add 'yt-dlp' to requirements.txt and redeploy."
        ) from e

    url = f"https://www.youtube.com/watch?v={video_id}"

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        # Try android client first (lightest IP filtering), then web as backup
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "web", "tv_embedded"],
            }
        },
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    subtitles = info.get("subtitles", {})      # manually uploaded captions
    auto_captions = info.get("automatic_captions", {})  # auto-generated

    def _find_json3_url(source: dict, lang_list: list[str]) -> str | None:
        """Return the first json3-format URL found for any language in lang_list."""
        for lang in lang_list:
            if lang in source:
                for fmt in source[lang]:
                    if fmt.get("ext") == "json3":
                        return fmt["url"]
        return None

    # Build a prioritised language search order:
    #   requested langs → all available manual langs → all available auto langs
    search_order = languages + list(subtitles.keys()) + list(auto_captions.keys())

    subtitle_url = (
        _find_json3_url(subtitles, search_order)
        or _find_json3_url(auto_captions, search_order)
    )

    if not subtitle_url:
        raise ValueError(
            f"yt-dlp found no subtitles/captions for video {video_id}. "
            "The video may not have any captions enabled."
        )

    # Fetch the actual subtitle content (plain GET — no auth needed)
    resp = req.get(
        subtitle_url,
        timeout=20,
        headers={
            # Match the android client UA that yt-dlp used to get the URL
            "User-Agent": (
                "com.google.android.youtube/19.09.37 "
                "(Linux; U; Android 11; Pixel 4 Build/RQ3A.210905.001)"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    resp.raise_for_status()

    segments = _parse_json3(resp.json())
    if not segments:
        raise ValueError(
            f"Subtitle file returned for {video_id} was empty."
        )
    return segments


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_transcript(
    video_id: str,
    languages: list[str] | None = None,
    proxy_config=None,
) -> list[dict]:
    """
    Fetch the transcript for a YouTube video.

    Automatic two-strategy fetch:
      1. youtube-transcript-api (fast; great on local dev)
      2. yt-dlp android/web client (cloud-compatible fallback; no signup needed)

    Args:
        video_id:     11-character YouTube video ID.
        languages:    Preferred language codes (default: English variants).
        proxy_config: Optional ProxyConfig from build_proxy_config().
                      Enhances Strategy 1 reliability on cloud; not needed for
                      Strategy 2.

    Returns:
        List of segment dicts: [{"text": str, "start": float, "duration": float}, ...]

    Raises:
        ValueError: If both strategies fail (with details from both errors).
    """
    langs = languages or ["en", "en-US", "en-GB"]

    # ── Strategy 1: youtube-transcript-api ───────────────────────────────────
    primary_exc: Exception | None = None
    try:
        result = _fetch_via_ytt(video_id, langs, proxy_config)
        logger.debug("Transcript fetched via youtube-transcript-api for %s", video_id)
        return result
    except Exception as exc:
        primary_exc = exc
        logger.warning(
            "youtube-transcript-api failed for %s (%s) — trying yt-dlp fallback…",
            video_id,
            type(exc).__name__,
        )

    # ── Strategy 2: yt-dlp android/web player client ─────────────────────────
    try:
        result = _fetch_via_ytdlp(video_id, langs)
        logger.info("Transcript fetched via yt-dlp fallback for %s", video_id)
        return result
    except Exception as fallback_exc:
        raise ValueError(
            f"Failed to fetch transcript for {video_id}.\n\n"
            f"  Strategy 1 (youtube-transcript-api): {primary_exc}\n"
            f"  Strategy 2 (yt-dlp android client):  {fallback_exc}\n\n"
            "Both methods were blocked. If running on a cloud platform, add a "
            "residential proxy via HTTP_PROXY / HTTPS_PROXY in your app secrets."
        ) from fallback_exc


def segments_to_full_text(segments: list[dict]) -> str:
    """Concatenate all segment texts into a single string."""
    return " ".join(seg["text"].strip() for seg in segments)


def get_video_metadata(video_id: str) -> dict:
    """Return lightweight metadata dict (URL only; title set by user in UI)."""
    return {
        "video_id": video_id,
        "video_url": f"https://www.youtube.com/watch?v={video_id}",
    }
