"""Streamlit UI entry point for finding exact dialogue frames in videos (Phase 7 / ADR-7).

Wires together the pipeline stages:
1. URL Validation & Video Resource Confirmation (ADR-2)
2. Video/Audio Download (ADR-2 / yt-dlp)
3. Speech Recognition with Forced Alignment (ADR-3 / WhisperX)
4. Hybrid Fuzzy + Semantic Dialogue Matching (ADR-4)
5. Ambiguity & Confidence Resolution (ADR-6)
6. Frame Localization & Extraction (ADR-5)
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

# Ensure project root is in sys.path when executed via `streamlit run src/app.py`
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st

from src.confidence import ResolutionResult, ResolutionState, resolve
from src.frame_localization import (
    FrameExtractionError,
    compute_frame_number,
    extract_frame,
)
from src.matching import Candidate, find_candidates
from src.transcription import transcribe
from src.url_validation import ValidationResult, VideoMetadata, validate_video_url


def download_video_or_audio(url: str, output_dir: str) -> Optional[str]:
    """Download video resource into output_dir using yt-dlp with fallback handling."""
    # Check local files first
    if os.path.isfile(url):
        return os.path.abspath(url)
    if url.startswith("file://"):
        local_target = url[7:]
        if os.path.isfile(local_target):
            return os.path.abspath(local_target)

    video_path = os.path.join(output_dir, "video.mp4")

    # Helper function to execute yt-dlp with optional impersonation
    def _run_ytdlp(use_impersonate: bool) -> Optional[str]:
        import yt_dlp

        ydl_opts = {
            "format": "bestvideo+bestaudio/bestvideo/best",
            "outtmpl": os.path.join(output_dir, "video.%(ext)s"),
            "merge_output_format": "mp4",
            "quiet": True,
            "no_warnings": True,
            "nocheckcertificate": True,
            "legacy_server_connect": True,
            "socket_timeout": 30,
            "retries": 10,
            "fragment_retries": 10,
        }
        if use_impersonate:
            ydl_opts["impersonate"] = "chrome"

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if hasattr(ydl, "prepare_filename") and info:
                fn = ydl.prepare_filename(info)
                possible_mp4 = os.path.splitext(fn)[0] + ".mp4"
                if os.path.exists(possible_mp4) and os.path.getsize(possible_mp4) > 0:
                    return possible_mp4
                if os.path.exists(fn) and os.path.getsize(fn) > 0:
                    return fn

            for f in Path(output_dir).glob("video.*"):
                if f.is_file() and f.stat().st_size > 0:
                    return str(f)
        return None

    # 1a. Try yt-dlp with impersonate
    try:
        result = _run_ytdlp(use_impersonate=True)
        if result:
            return result
    except Exception:
        pass

    # 1b. Fallback: Try yt-dlp standard
    try:
        result = _run_ytdlp(use_impersonate=False)
        if result:
            return result
    except Exception as exc:
        st.warning(f"yt-dlp download notice: {exc}")

    # 2. Fallback for direct media links
    direct_video_extensions = (".mp4", ".mkv", ".webm", ".mov", ".avi")
    if url.lower().endswith(direct_video_extensions):
        try:
            import requests

            resp = requests.get(url, stream=True, timeout=15)
            if resp.status_code == 200:
                with open(video_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
                    return video_path
        except Exception as exc:
            st.warning(f"HTTP GET download notice: {exc}")

    return None


def extract_audio_from_video(video_path: str, output_dir: str) -> str:
    """Extract a 16kHz mono PCM WAV file from the video container for WhisperX."""
    audio_path = os.path.join(output_dir, "audio.wav")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        audio_path,
    ]
    res = subprocess.run(cmd, capture_output=True, check=False)
    if res.returncode == 0 and os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
        return audio_path
    return video_path


def render_results(
    resolution: ResolutionResult,
    video_path: str,
    fps: float,
    output_dir: str,
) -> None:
    """Render the final outcome matching one of the three ADR-6 ResolutionStates."""
    st.divider()
    st.header("Search Results")

    if resolution.state == ResolutionState.CONFIDENT:
        candidate: Candidate = resolution.primary  # type: ignore[assignment]
        st.success(f"Confident Match Found: \"{candidate.text}\"")

        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Match Details")
            st.markdown(f"**Matched Dialogue:** {candidate.text}")
            st.markdown(
                f"**Timestamp Range:** {candidate.start_sec:.2f}s – {candidate.end_sec:.2f}s"
            )
            frame_number = compute_frame_number(candidate.start_sec, fps)
            st.markdown(f"**Frame Number:** `{frame_number}` (at {fps:.2f} fps)")
            st.markdown(
                f"**Combined Confidence Score:** `{candidate.combined_score:.3f}`"
            )
            st.markdown(f"- *Semantic Similarity:* `{candidate.semantic_score:.3f}`")
            st.markdown(f"- *Fuzzy Lexical Score:* `{candidate.fuzzy_score:.3f}`")

        with col2:
            st.subheader("Extracted Frame")
            frame_path = os.path.join(output_dir, f"frame_{frame_number}.png")
            try:
                extracted = extract_frame(video_path, candidate.start_sec, frame_path)
                if extracted and os.path.exists(extracted):
                    st.image(
                        extracted,
                        caption=f"Frame #{frame_number} at {candidate.start_sec:.2f}s",
                        use_container_width=True,
                    )
            except Exception as exc:
                st.warning(f"Could not extract frame image: {exc}")

    elif resolution.state == ResolutionState.AMBIGUOUS:
        st.warning("Ambiguous match: Multiple close matches were found.")
        st.markdown(
            f"Found **{len(resolution.alternatives)}** candidate matches within margin threshold:"
        )

        for idx, cand in enumerate(resolution.alternatives, start=1):
            frame_num = compute_frame_number(cand.start_sec, fps)
            st.markdown("---")
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown(f"### Candidate #{idx}: \"{cand.text}\"")
                st.markdown(
                    f"**Timestamp Range:** {cand.start_sec:.2f}s – {cand.end_sec:.2f}s"
                )
                st.markdown(f"**Frame Number:** `{frame_num}` (at {fps:.2f} fps)")
                st.markdown(
                    f"**Combined Score:** `{cand.combined_score:.3f}` "
                    f"(Semantic: {cand.semantic_score:.3f}, Fuzzy: {cand.fuzzy_score:.3f})"
                )
            with col2:
                frame_path = os.path.join(output_dir, f"candidate_{idx}_{frame_num}.png")
                try:
                    extracted = extract_frame(video_path, cand.start_sec, frame_path)
                    if extracted and os.path.exists(extracted):
                        st.image(
                            extracted,
                            caption=f"Candidate #{idx} - Frame #{frame_num} at {cand.start_sec:.2f}s",
                            use_container_width=True,
                        )
                except Exception as exc:
                    st.warning(f"Could not extract frame image for candidate #{idx}: {exc}")

    elif resolution.state == ResolutionState.NOT_FOUND:
        st.error("No confident match found for the target dialogue.")
        if resolution.primary is not None:
            cand = resolution.primary
            frame_num = compute_frame_number(cand.start_sec, fps)
            st.info(
                f"Best-effort candidate (low confidence): \"{cand.text}\" "
                f"(Score: {cand.combined_score:.3f})"
            )
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown(f"**Dialogue:** {cand.text}")
                st.markdown(
                    f"**Timestamp Range:** {cand.start_sec:.2f}s – {cand.end_sec:.2f}s"
                )
                st.markdown(f"**Frame Number:** `{frame_num}` (at {fps:.2f} fps)")
                st.markdown(
                    f"**Combined Score:** `{cand.combined_score:.3f}` (Below floor threshold)"
                )
            with col2:
                frame_path = os.path.join(output_dir, f"best_effort_{frame_num}.png")
                try:
                    extracted = extract_frame(video_path, cand.start_sec, frame_path)
                    if extracted and os.path.exists(extracted):
                        st.image(
                            extracted,
                            caption=f"Best-effort Frame #{frame_num} at {cand.start_sec:.2f}s",
                            use_container_width=True,
                        )
                except Exception:
                    pass
        else:
            st.info("No dialogue candidates were identified in the video transcript.")


def run_pipeline(url: str, dialogue: str) -> None:
    """Execute the full end-to-end staged pipeline with st.status progress feedback."""
    with st.status("Processing video pipeline...", expanded=True) as status:
        # Stage 1: Validate URL
        st.write("🔍 **Stage 1:** Validating video URL...")
        val_result: ValidationResult = validate_video_url(url)
        if not val_result.is_valid:
            status.update(label="Validation Failed", state="error", expanded=False)
            st.error(val_result.error_message or "Please enter a valid http(s) video URL.")
            return

        fps: float = 30.0
        if val_result.metadata and val_result.metadata.fps:
            fps = float(val_result.metadata.fps)

        # Create persistent downloads directory in project root
        downloads_dir = Path(_PROJECT_ROOT) / "downloads"
        downloads_dir.mkdir(parents=True, exist_ok=True)
        target_dir = str(downloads_dir)

        # Stage 2: Fetch / Download
        st.write(f"📥 **Stage 2:** Fetching video resource into `{downloads_dir.name}/`...")
        video_path = download_video_or_audio(url, target_dir)
        if not video_path or not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
            status.update(label="Download Failed", state="error", expanded=False)
            st.error("Failed to download video file or the target resource is empty.")
            return

        file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        st.write(f"✅ Video downloaded: `{os.path.basename(video_path)}` ({file_size_mb:.2f} MB)")

        # Stage 3: Transcribe
        st.write("🎙️ **Stage 3:** Transcribing audio with WhisperX forced alignment...")
        audio_path = extract_audio_from_video(video_path, target_dir)
        words = transcribe(audio_path)

        # Stage 4: Match
        st.write("🔎 **Stage 4:** Matching candidate dialogue spans...")
        candidates = find_candidates(words, dialogue)

        # Stage 5: Confidence Resolution & Frame Localization
        st.write("🎯 **Stage 5:** Resolving confidence and localizing frames...")
        resolution = resolve(candidates)

        status.update(label="Processing Complete!", state="complete", expanded=False)

        # Render results
        render_results(resolution, video_path, fps, target_dir)


def main() -> None:
    """Streamlit application UI."""
    st.set_page_config(
        page_title="Video Dialogue Frame Finder",
        page_icon="🎬",
        layout="wide",
    )

    st.title("🎬 Video Dialogue Frame Finder")
    st.markdown(
        "Locate the exact timestamp and video frame where a spoken dialogue appears in an online video."
    )

    video_url = st.text_input(
        "Video URL",
        placeholder="https://example.com/video or https://ok.ru/video/...",
        help="Direct URL to a video or supported streaming platform resource.",
    )

    target_dialogue = st.text_input(
        "Target Dialogue Phrase",
        placeholder="e.g., my mind rebels at stagnation",
        help="Dialogue phrase to search for (paraphrases and approximate quotes supported).",
    )

    find_button = st.button("Locate Frame", type="primary")

    if find_button:
        if not video_url.strip():
            st.error("Please enter a valid http(s) video URL.")
        elif not target_dialogue.strip():
            st.error("Please enter a target dialogue phrase.")
        else:
            run_pipeline(video_url.strip(), target_dialogue.strip())


if __name__ == "__main__":
    main()