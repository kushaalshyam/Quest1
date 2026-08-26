# 🎬 Video Dialogue Frame Finder

An AI-powered video processing application that pinpoints the exact timestamp and visual frame where a spoken dialogue phrase appears in online streaming URLs or direct video files.

---

## 📌 Features

* **Multi-Source Validation:** Supports streaming platform URLs (`yt-dlp`) and direct media files with HTTP HEAD validation.
* **Forced Phoneme Alignment:** Utilizes `WhisperX` and `Pyannote` for word-level speech recognition timestamps.
* **Hybrid Dialogue Matching:** Combines fuzzy lexical matching (`rapidfuzz`) with semantic embedding similarity (`sentence-transformers`) to handle paraphrases, partial quotes, and ASR mis-transcriptions.
* **Frame-Accurate Extraction:** Computes exact frame numbers and extracts PNG snapshot images using frame-accurate `FFmpeg` seeks.
* **Staged Streamlit Interface:** Real-time visual feedback tracking pipeline execution from validation to frame localization.

---

## 📁 Project Structure

```text
Quest1/
├── .gitignore
├── README.md
├── requirements.txt
├── downloads/
└── src/
    ├── app.py                  # Streamlit application UI & pipeline orchestrator
    ├── confidence.py           # Confidence scoring & ambiguity resolution logic
    ├── frame_localization.py   # Frame index calculation & FFmpeg extraction
    ├── matching.py             # Hybrid fuzzy + semantic dialogue search engine
    ├── transcription.py        # WhisperX transcription & forced alignment pipeline
    └── url_validation.py       # Layered URL validation & metadata extraction
```

---

## ⚙️ Prerequisites

1. **Python:** Version 3.10 to 3.13
2. **FFmpeg:** Required on system `PATH` for video/audio decoding and frame extraction.

### Installing FFmpeg

* **macOS (via Homebrew):**
  ```bash
  brew install ffmpeg
  ```

* **Windows (via winget or Chocolatey):**
  ```cmd
  winget install ffmpeg
  ```

  *Or via Chocolatey:*
  ```cmd
  choco install ffmpeg
  ```

  *(Note: Restart your terminal after installing on Windows so the system `PATH` updates.)*

---

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Quest1
```

### 2. Create & Activate Virtual Environment

* **macOS / Linux:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

* **Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```

* **Windows (Command Prompt):**
  ```cmd
  python -m venv .venv
  .\.venv\Scripts\activate.bat
  ```

### 3. Install Dependencies

Create a `requirements.txt` file in your project root with the following contents:

```text
streamlit
whisperx
torch
torchaudio
yt-dlp
curl_cffi
rapidfuzz
sentence-transformers
requests
```

Then run:

```bash
pip install -r requirements.txt
```

### 4. Patch Metadata Dependencies (WhisperX)

To resolve `huggingface-hub` version constraint conflicts between `whisperx` and `transformers`:

```bash
pip install "huggingface-hub>=1.5.0,<2.0.0"
pip install whisperx --no-deps --force-reinstall
```

*(Optional, for Windows users with NVIDIA GPUs)* — enable CUDA hardware acceleration for faster WhisperX inference:

```cmd
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## 🖥️ Usage

Launch the Streamlit web application:

```bash
streamlit run src/app.py
```

1. Open `http://localhost:8501` in your browser.
2. Enter a video URL (e.g., YouTube, OK.ru, or a direct `.mp4` link).
3. Enter the spoken target dialogue phrase (exact quote or approximate paraphrase).
4. Click **Locate Frame** to run the pipeline and view the extracted snapshot frame and confidence metrics.

---

## 📝 Configuration & Notes

* **Local Output Directory:** Downloaded `.mp4` video files, extracted `.wav` audio tracks, and localized frame `.png` files are stored in the project's `downloads/` directory.
* **Network Blocking:** When processing URLs from platforms with strict anti-bot measures, `curl_cffi` browser impersonation is automatically invoked by `yt-dlp`. If a TCP connection reset occurs due to network/ISP restrictions, switch to a non-restricted network or toggle a VPN.
