"""URL validation and video-resource confirmation (ADR-2).

Layered checks: urlparse (reject malformed input with no network I/O) →
yt-dlp extract_info(download=False) with automatic fallback → HTTP HEAD fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse


@dataclass
class VideoMetadata:
    duration_sec: Optional[float]
    fps: Optional[float]
    title: str
    source_extractor: str  # "ytdlp" or "http_head_fallback"


@dataclass
class ValidationResult:
    is_valid: bool
    metadata: Optional[VideoMetadata]
    error_message: Optional[str]  # populated when is_valid is False


def _is_syntactically_valid_http_url(url: str) -> bool:
    if not url or not str(url).strip():
        return False
    parsed = urlparse(url)
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def _content_type_is_video(headers) -> bool:
    raw = ""
    if headers is not None:
        raw = headers.get("Content-Type") or headers.get("content-type") or ""
    media_type = str(raw).split(";", 1)[0].strip().lower()
    return media_type.startswith("video/")


def _invalid(message: str) -> ValidationResult:
    return ValidationResult(is_valid=False, metadata=None, error_message=message)


def _try_ytdlp_extract(url: str, impersonate: bool) -> Optional[VideoMetadata]:
    import yt_dlp

    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "no_warnings": True,
        "nocheckcertificate": True,
        "legacy_server_connect": True,
        "socket_timeout": 20,
        "retries": 10,
    }
    if impersonate:
        ydl_opts["impersonate"] = "chrome"

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        if info:
            return VideoMetadata(
                duration_sec=info.get("duration"),
                fps=info.get("fps"),
                title=info.get("title") or "",
                source_extractor="ytdlp",
            )
    return None


def validate_video_url(url: str) -> ValidationResult:
    if not _is_syntactically_valid_http_url(url):
        return _invalid("Please enter a valid http(s) video URL.")

    # 1a. Try yt-dlp with browser impersonation
    try:
        meta = _try_ytdlp_extract(url, impersonate=True)
        if meta:
            return ValidationResult(is_valid=True, metadata=meta, error_message=None)
    except Exception:
        pass

    # 1b. Fallback: Try yt-dlp standard (in case curl_cffi is missing)
    try:
        meta = _try_ytdlp_extract(url, impersonate=False)
        if meta:
            return ValidationResult(is_valid=True, metadata=meta, error_message=None)
    except Exception:
        pass

    # 2. HTTP HEAD Fallback for direct media file links (.mp4, .mov, etc.)
    try:
        import requests

        response = requests.head(url, allow_redirects=True, timeout=10)
        if _content_type_is_video(getattr(response, "headers", None)):
            return ValidationResult(
                is_valid=True,
                metadata=VideoMetadata(
                    duration_sec=None,
                    fps=None,
                    title="",
                    source_extractor="http_head_fallback",
                ),
                error_message=None,
            )
        return _invalid(
            "This URL is not a supported video resource. "
            "Please re-enter a valid or supported video URL."
        )
    except Exception:
        return _invalid(
            "Could not confirm this URL as a video resource. "
            "Please re-enter a valid video URL."
        )