"""Frame localization: map a timestamp to a frame index under CFR (ADR-5)."""

from __future__ import annotations

import subprocess


class FrameExtractionError(Exception):
    """Raised when ffmpeg fails to extract a frame at the requested timestamp."""


def compute_frame_number(timestamp_sec: float, fps: float) -> int:
    """Return the nearest frame index for timestamp_sec at constant fps.

    Uses Python's round() on timestamp_sec * fps. Assumes constant frame
    rate; VFR sources may drift (documented in ADR-5).
    """
    return round(timestamp_sec * fps)


def extract_frame(
    video_path: str, timestamp_sec: float, output_image_path: str
) -> str:
    """Extract a single frame at timestamp_sec via ffmpeg accurate seek (ADR-5 B).

    ``-ss`` is an output option (after ``-i``) so ffmpeg decodes to the exact
    timestamp rather than the nearest keyframe. Returns output_image_path on
    success; raises FrameExtractionError on non-zero exit or subprocess failure.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-ss",
        str(timestamp_sec),
        "-frames:v",
        "1",
        output_image_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True)
    except Exception as exc:
        raise FrameExtractionError(
            f"Failed to extract frame at {timestamp_sec}s from {video_path}"
        ) from exc

    if result.returncode != 0:
        stderr = result.stderr
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        raise FrameExtractionError(
            f"ffmpeg exited {result.returncode} extracting frame at "
            f"{timestamp_sec}s from {video_path}: {stderr}"
        )

    return output_image_path

