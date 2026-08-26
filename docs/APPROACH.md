# APPROACH.md

## Find the Exact Frame Where a Dialogue Appears in a Video

---

## 1. Problem Interpretation & Scope

The problem statement asks the system to locate the frame in which an "on-screen dialogue" first appears, given a video URL and a target dialogue phrase.

**Key interpretation decision:** The phrase "on-screen dialogue" is interpreted as *dialogue delivered during video playback* (i.e., spoken dialogue, audible in the audio track), not literal burned-in/overlay text on the video frame. This interpretation is based on direct inspection of the provided sample video (`https://ok.ru/video/248244667877`), which contains no burned-in or overlay text corresponding to the target line "My mind rebels at stagnation" — the line is spoken.

Given this, an **ASR (Automatic Speech Recognition)-based pipeline** is the primary approach for v1. A frame-OCR-based pipeline (for videos with genuine burned-in subtitles/text overlays) is explicitly named as **out-of-scope / future extended scope**, since the given sample does not require it and the problem statement's note that a different video may be used in evaluation is treated as a known, documented limitation rather than an unaddressed gap.

### Out of Scope (v1)
- OCR-based extraction of literal on-screen text overlays.
- Support for private/authenticated video URLs.
- Multi-dialogue-per-run search (system is designed for one target dialogue per run).
- Hard duration cap on input video — any length is supported, with the explicit caveat that processing time scales with video duration (validated on a 50-minute sample; not stress-tested at multi-hour scale).

---

## 2. Architecture Overview

```
User Input (URL + target dialogue text)
        │
        ▼
[1] URL Validation & Video Resource Confirmation
        │  (yt-dlp extract_info → metadata: duration, fps, formats)
        ▼
[2] Audio Extraction (ffmpeg)
        │
        ▼
[3] Transcription (WhisperX — word-level, forced-aligned timestamps)
        │
        ▼
[4] Dialogue Matching (hybrid fuzzy + semantic similarity over transcript windows)
        │
        ▼
[5] Ambiguity / Confidence Resolution (floor threshold + margin check)
        │
        ▼
[6] Frame Localization & Extraction (timestamp × fps → frame number; ffmpeg accurate seek → image)
        │
        ▼
[7] Streamlit UI: staged progress + final result (confident / ambiguous / not found)
```

---

## 3. Design Decisions (ADR Log)

Each section below follows: **Context → Options Considered → Decision → Consequences.**

---

### ADR-1: Scope of "On-Screen Dialogue" — ASR vs OCR vs Hybrid

**Context**
The literal phrase "on-screen dialogue" could mean either spoken dialogue or literal text overlay. The provided sample video contains no text overlay for the target line.

**Options Considered**
| Option | Description |
|---|---|
| ASR-only | Transcribe audio, match dialogue text to spoken words |
| OCR-only | Sample video frames, run OCR, match text region-by-region |
| Hybrid (ASR primary + OCR fallback) | Run ASR as primary path; fall back to sparse-frame OCR if ASR match confidence is low, or if the target dialogue never appears in the transcript |

**Decision:** ASR-only for v1, based on direct inspection of the sample input. Hybrid is documented as the natural extended-scope evolution, not implemented now.

**Consequences**
- v1 will not correctly handle videos where the dialogue is genuine burned-in text with no corresponding audio.
- This limitation is explicitly documented rather than silently assumed, so it can be defended and extended on request.

---

### ADR-2: URL Validation & Video-Resource Confirmation

**Context**
Must confirm, without manual inspection, that a given input string is (a) a syntactically valid URL, and (b) a reachable, actual video resource.

**Options Considered**
| Option | Notes |
|---|---|
| ICMP ping | Doesn't confirm HTTP reachability or content type; often blocked by hosts → false negatives |
| `urlparse` syntax check only | Fast, but confirms nothing about reachability/content |
| HTTP `HEAD` request | Lightweight, but video-hosting *pages* (YouTube, ok.ru) return `text/html`, not `video/*`, since the actual video is behind JS/streaming manifests |
| `yt-dlp.extract_info(download=False)` | Confirms extractor support, resource existence, and returns metadata (duration, fps, formats, title) in one call |
| Hybrid: `urlparse` → yt-dlp → HTTP HEAD fallback | Layered validation covering both platform URLs and direct video file links |

**Decision:** Hybrid layered validation.
1. `urlparse` rejects malformed input immediately.
2. `yt-dlp.extract_info(download=False)` is the primary validation and metadata-fetch step (also supplies fps needed later for frame-number calculation).
3. If yt-dlp cannot extract (unsupported site), fall back to HTTP `HEAD` and check `Content-Type` starts with `video/`.
4. If all fail, prompt the user with a clear message to re-enter a valid/supported video URL.

**Consequences**
- yt-dlp is already required later for audio download, so this adds no new dependency.
- Validation and metadata retrieval are combined into a single pipeline step.
- Adds ~1–3s upfront latency, acceptable as a one-time gate.

---

### ADR-3: Transcription Engine

**Context**
Need timestamped transcription precise enough to localize a short dialogue line to an individual frame, not just a coarse segment.

**Options Considered**
| Option | Notes |
|---|---|
| `openai-whisper` | Segment-level timestamps by default (~5–30s chunks); word-level flag exists but alignment is approximate, not forced |
| `faster-whisper` | Same timestamp mechanism as above, faster runtime, doesn't fix precision |
| `WhisperX` | Word-level timestamps via forced phoneme alignment (wav2vec2-based) — significantly tighter per-word timing |
| Cloud ASR (AssemblyAI / Deepgram / Google STT) | Precise, managed, but paid, API-dependent, and sends video/audio to a third party |

**Decision:** WhisperX. The distinguishing factor is not chunk size (vanilla Whisper does not literally sample at fixed 5s intervals — it decodes in ~30s windows and emits variable-length segments based on pause/silence heuristics) but the **method of deriving word-level timing**. Vanilla Whisper's optional word-level timestamps are inferred from the model's internal cross-attention weights during decoding — a byproduct of the language model, not a direct measurement against the audio signal — and are known to drift, especially over longer segments. WhisperX decouples this: it takes Whisper's transcript text and runs a separate forced-alignment pass using a phoneme-recognition model (wav2vec2-based CTC) that aligns each word directly to the audio waveform at near-frame granularity, independent of Whisper's attention weights. This audio-grounded alignment is what makes frame-accurate localization (`timestamp × fps`) meaningful rather than a segment-level approximation. Cloud ASR rejected to keep the solution self-contained, offline, and free of third-party data dependency and per-request cost.

**Consequences**
- Adds an alignment-model dependency (wav2vec2) beyond base Whisper.
- Directly supports defensible frame-number accuracy.
- Fully local — no API key or external service dependency.

---

### ADR-4: Dialogue Matching Strategy

**Context**
The user-provided dialogue may be an exact quote or a paraphrase (e.g., "my mind is against stagnation" vs. target "my mind rebels at stagnation"). Need a matching approach robust to both, and a numeric confidence score for downstream ambiguity handling.

**Sub-decision: Candidate Generation — Why Sliding-Window-by-Word-Count**

Localizing an exact timestamp requires per-span candidates with their own start/end times, not a single global similarity score against the whole transcript — this constraint rules out any approach that only answers "is the phrase present," and shapes the rest of this sub-decision. A **span** here means a contiguous slice of the word-level transcript (a start index and end index into the `WordTiming` list); its start/end timestamps come directly from its first and last word's timings.

| Option | Notes |
|---|---|
| Match against ASR's natural segment/sentence boundaries | Segment boundaries (based on pause/punctuation heuristics) are unreliable for this purpose — a short line can span two segments if there's a mid-sentence pause, or one segment can contain multiple distinct lines. Matching accuracy would be hostage to a heuristic outside our control |
| Fixed time-based windows (e.g. every 3 seconds) | Speaking rate varies by speaker and delivery, so a fixed time window captures a variable number of words — sometimes too few to ever match, sometimes too many. Since the actual comparison is text-based, window size should be defined in the same unit being compared (words), not time |
| Sliding window over word-level transcript, sized to target word count ± `window_slack`, combined with a fuzzy pre-filter to bound downstream cost | Independent of ASR segmentation; window units (words) match the comparison units (words); simple enough to specify as an exact, testable contract; brute-force cost is absorbed by the fuzzy pre-filter (`prefilter_top_n`) rather than needing an algorithmically cleverer search |

**Decision:** Sliding window over the word-level transcript. N is `len(target_dialogue.split())` (whitespace split of the target as given; punctuation attached to a token does not create an extra window size). Generate candidate spans at every window size in the inclusive range `[max(1, N − window_slack), N + window_slack]` (default `window_slack = 2`; lower bound clamped at 1 so a short target cannot produce empty/zero-length windows), sliding one word at a time across the full transcript at each size. This produces many overlapping candidate spans, each with its own start/end timestamp — exactly what's needed for the localization requirement — while staying simple enough to unit-test deterministically (see `docs/TEST_STRATEGY.md` T3.6a–T3.6e, which pin down that both ends of the inclusive range are actually reached, that the range does not silently extend past `N + window_slack`, that `window_slack=0` collapses generation to fixed-size windows rather than the parameter being ignored, and that a raw `N − window_slack < 1` is clamped to 1).

**Options Considered (fuzzy vs. semantic scoring)**
| Option | Notes |
|---|---|
| Pure lexical fuzzy (rapidfuzz `token_sort_ratio` / `partial_ratio`) | Strong on exact/near-exact matches; weak on paraphrases with low token overlap |
| Pure semantic (sentence-transformer embeddings + cosine similarity) | Strong on paraphrase/meaning matches; can occasionally over-match on topical similarity |
| Weighted hybrid (normalized combination of both scores) | Combines strengths of both signals |

**Why both signals are necessary (not redundant):** Semantic similarity alone solves the paraphrase case, but it has an independent, mirror-image failure mode — similar *meaning* does not guarantee it is the *same line*. Specifically:
- **Topically-similar false positives:** a long video likely contains several lines expressing similar sentiment to the target (e.g. "I can't stand doing nothing" vs. "my mind rebels at stagnation"). A semantic model can score several of these nearly as high as the true match, since it captures theme/sentiment rather than the specific phrasing. Fuzzy score favors the candidate with genuine lexical overlap, which is usually the actual line.
- **Short-text embedding noise:** sentence embedding models are trained mostly on longer text; on short dialogue-length phrases (4–6 words) the embedding space is noisier, and unrelated short phrases can land closer together than intuition suggests.
- **ASR transcription noise:** if Whisper mis-hears a word phonetically (e.g. "rebels" → "reveals"), the transcript is lexically close to the true line but the *meaning* may shift — semantic similarity can drop for the correct span while fuzzy similarity survives, since character/token overlap is more robust to this kind of noise. This matters because matching is performed against a noisy ASR **transcript**, not ground-truth text.

In short: semantic similarity covers "different words, same meaning" (paraphrase); fuzzy similarity covers "same/similar words, but the semantic model may not discriminate well, or the transcript itself is noisy." The two signals compensate for largely independent failure modes, which is the actual justification for combining them rather than a general "hybrid is more robust" claim.

**Decision:** Weighted hybrid, with fuzzy matching as a cheap pre-filter and semantic scoring reserved for the strongest lexical candidates:
1. Slide a window over the WhisperX word-level transcript (window sizes ≈ user input length, ±2 words) to build all candidate spans with aggregate start/end timestamps.
2. Compute a fuzzy score (rapidfuzz) for **every** candidate span — this is cheap enough to run exhaustively even on a 50-minute transcript's worth of windows.
3. **Pre-filter:** take only the top-N candidates by fuzzy score (e.g. N=20, tunable) forward to semantic scoring, rather than embedding every window. This is both a performance optimization (avoids running the embedding model over the entire transcript) and a practical acknowledgment that a true match will virtually always retain at least moderate lexical overlap — even a loose paraphrase shares some words with the target.
4. Compute a semantic score (lightweight sentence-transformer, e.g. `all-MiniLM-L6-v2`) only for those top-N candidates.
5. Combine fuzzy and semantic into a single normalized confidence score (0–1) for the top-N set: `combined_score = 0.6 * semantic_score + 0.4 * fuzzy_score` (semantic weighted higher because paraphrase tolerance is an explicit requirement).
6. Rank the top-N candidates by combined score; all other candidates are implicitly scored as fuzzy-only and excluded (they did not clear the pre-filter).

**Consequences**
- Adds a sentence-transformer model dependency and inference step, but its cost is bounded to N candidates rather than the full transcript window count — a meaningful performance win on long videos.
- Introduces a new tunable parameter (pre-filter size N), documented alongside window slack as a non-hardcoded constant.
- Edge case: a genuine paraphrase with near-zero lexical overlap could theoretically fail to make the top-N fuzzy pre-filter and never reach semantic scoring. This is a documented, accepted trade-off — N is set generously (20) specifically to make this scenario unlikely in practice, and is called out here rather than silently assumed.
- Sliding-window granularity remains a tunable parameter, not a hardcoded constant.
- The resulting confidence score directly feeds ambiguity handling (ADR-6).

---

### ADR-5: Frame Localization & Extraction

**Context**
Given a word-level timestamp for the matched dialogue, need to (a) compute a frame number, and (b) extract that frame as an image.

**Sub-decision A — Frame number calculation**
| Option | Notes |
|---|---|
| `frame_number = round(timestamp × fps)` | Simple, assumes constant frame rate (CFR) |
| Walk actual per-frame PTS via ffprobe/PyAV | Handles variable frame rate (VFR) exactly, adds parsing complexity |

**Decision:** Arithmetic approach (`timestamp × fps`), with an explicitly documented assumption of constant frame rate. VFR sources may introduce minor (sub-second) frame-number drift; this is a stated, acceptable simplification for v1, and PTS-walking is the documented next step for extended scope.

**Sub-decision B — Frame extraction method**
| Option | Notes |
|---|---|
| OpenCV (`cv2.VideoCapture` seek/iterate) | `.set(CAP_PROP_POS_FRAMES)` seeking is unreliable on compressed video (seeks to nearest keyframe); sequential iteration is accurate but slow on long video |
| `ffmpeg` accurate timestamp seek + single-frame extract | Fast, timestamp-accurate seeking is a first-class ffmpeg feature, consistent across codecs |

**Decision:** ffmpeg accurate seek by timestamp. ffmpeg is already a pipeline dependency (used for audio extraction), so this adds no new tooling. The displayed frame number is a **derived value** (from timestamp × fps under the CFR assumption), not independently verified against the container's frame index — this distinction is stated plainly rather than implied as exact.

**Consequences**
- One shared ffmpeg dependency for both audio and frame extraction.
- Known, stated limitation on frame-number accuracy for VFR sources.

---

### ADR-6: Ambiguity / Confidence Handling

**Context**
Need a principled policy for when to return a single confident answer, when to present multiple candidates, and when to report no match — directly addressing an explicit evaluation criterion.

**Options Considered**
| Option | Notes |
|---|---|
| Always return top-1, no threshold | Simple but silently wrong on ambiguous or absent matches |
| Single absolute threshold | Doesn't distinguish "one clear best match at low score" from "several near-identical matches at similar score" |
| Absolute floor threshold + relative margin between top-1 and top-2 | Distinguishes "no good match" from "multiple competing matches" — two distinct real failure modes |

**Decision:** Two-signal policy producing three output states:
1. **Confident** — top-1 score ≥ floor threshold (e.g. 0.75) AND margin over top-2 ≥ margin threshold (e.g. 0.10) → return a single result (timestamp, frame, text, image).
2. **Ambiguous** — top-1 clears the floor but margin is insufficient → return top-k (e.g. top 3) candidates, each with its own timestamp/frame/text/score, explicitly labeled as multiple close matches.
3. **Not Found** — top-1 does not clear the floor → report no confident match, optionally surfacing the best-effort top candidate labeled clearly as low-confidence.

Threshold values are explicitly documented as **tunable parameters**, validated empirically against the sample video rather than derived analytically.

**Consequences**
- Directly and concretely satisfies the "how do you handle ambiguity/uncertainty" evaluation criterion.
- Requires the UI to support three distinct result-rendering states.
- Threshold values are a known, stated soft spot that may need retuning for a different evaluation video — the *policy* remains sound regardless.

---

### ADR-7: Frontend UX Flow (Streamlit)

**Context**
Frontend must accept a URL and target dialogue text, run a potentially multi-minute pipeline, and present one of the three result states from ADR-6, without the user needing to inspect the video.

**Options Considered**
| Option | Notes |
|---|---|
| Single-page, single-shot form | Simple, but gives no feedback during a long-running pipeline |
| Multi-step wizard (separate pages per stage) | Overkill for two inputs and adds navigation complexity |
| Single page with staged progress feedback (`st.status` per pipeline stage) | Keeps simplicity, fixes silent-wait problem, gives per-stage failure visibility |

**Decision:** Single-page form with staged progress feedback: validate → fetch/download → transcribe → match → localize & extract frame. Final result renders as one of the three ADR-6 states.

**Consequences**
- Failure at any stage produces a specific, actionable message (e.g., failure at "fetch" vs. failure at "transcribe") rather than a generic error at the end.
- No additional session-state/navigation complexity in Streamlit.

---

## 4. Non-Functional Requirements Addressed

- **Robustness to quality/resolution/fps variation:** fps is read from source metadata (yt-dlp) rather than assumed; WhisperX transcription is resolution-independent (audio-based).
- **No manual video inspection required:** entire pipeline is automated end-to-end from URL + dialogue text input to final frame/timestamp output.
- **Paraphrase tolerance:** addressed directly by the hybrid fuzzy + semantic matching strategy (ADR-4).
- **Explainability:** every result carries a numeric confidence score and a stated basis for how it was derived (ADR-4, ADR-6).
- **Video length:** no hard cap; processing time scales with duration. Validated against the ~50-minute sample video; not stress-tested at multi-hour scale — stated as a known scaling consideration, not a hard limit.

## 5. Known Limitations / Future Scope

- OCR-based extraction for videos with genuine burned-in/overlay text (ADR-1).
- Variable-frame-rate (VFR) exact frame-index resolution via container PTS walking (ADR-5).
- Threshold auto-tuning / calibration against a broader labeled dataset (ADR-6).
- Support for authenticated/private video sources.