"""Audio transcription with word-level forced phoneme alignment (ADR-3).

Uses WhisperX:
1. Load audio and transcribe using Whisper ASR.
2. Run forced phoneme alignment (wav2vec2 CTC via whisperx.align) to get
   precise word-level timestamps.
3. Return a list of WordTiming dataclass instances.
"""

from __future__ import annotations

from typing import List, Optional

import whisperx

from src.matching import WordTiming


def transcribe(
    audio_path: str,
    model_name: str = "base",
    device: Optional[str] = None,
    batch_size: int = 16,
    compute_type: Optional[str] = None,
    language: Optional[str] = None,
) -> List[WordTiming]:
    """Transcribe an audio file and align words using WhisperX.

    Args:
        audio_path: Path to the audio file.
        model_name: Whisper model size/name (default: "base").
        device: Device to run inference on ("cuda", "cpu", etc.). Defaults to auto-detection.
        batch_size: Batch size for transcription (default: 16).
        compute_type: Computation type (e.g. "float16", "int8", "float32"). Defaults based on device.
        language: Language code if known (e.g. "en"). Defaults to auto-detected language.

    Returns:
        List of WordTiming instances with word, start_sec, and end_sec.
    """
    if device is None:
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    if compute_type is None:
        compute_type = "float16" if device == "cuda" else "int8"

    audio = whisperx.load_audio(audio_path)

    # 1. Transcribe with Whisper
    model = whisperx.load_model(
        model_name,
        device=device,
        compute_type=compute_type,
        language=language,
    )
    transcribe_result = model.transcribe(audio, batch_size=batch_size)

    segments = transcribe_result.get("segments", [])
    if not segments:
        return []

    # 2. Align whisper output with forced phoneme alignment
    detected_language = language or transcribe_result.get("language", "en")
    align_model, align_metadata = whisperx.load_align_model(
        language_code=detected_language,
        device=device,
    )
    aligned_result = whisperx.align(
        segments,
        align_model,
        align_metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    # 3. Extract word-level timings
    raw_words = aligned_result.get("word_segments")
    if not raw_words:
        raw_words = [
            w
            for segment in aligned_result.get("segments", [])
            for w in segment.get("words", [])
        ]

    word_timings: List[WordTiming] = []
    for w in raw_words:
        start = w.get("start")
        end = w.get("end")
        if start is not None and end is not None:
            word_timings.append(
                WordTiming(
                    word=str(w.get("word", "")),
                    start_sec=float(start),
                    end_sec=float(end),
                )
            )

    return word_timings
