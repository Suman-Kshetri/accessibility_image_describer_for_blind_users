# Accessibility Image Describer (Capstone — Lab 7 extension)

Real-time camera or upload → image caption → optional translation →
speech, built for blind/low-vision users. Runs entirely on CPU — no
GPU or external API needed.

## Pipeline

```
Camera/Upload → BLIP caption (tuned for detail: beam search, longer
              min/max length, repetition penalty)
              → EasyOCR text detection (optional, reads signs/labels)
              → GoogleTranslator (optional) → gTTS → audio
```

## Files

- `part_a_image_description.py` — Part A, standalone: BLIP captioning
  tuned with beam search and longer output for richer descriptions
  directly from the vision model, no separate rewriting model needed.
  Test with: `python part_a_image_description.py path/to/image.jpg`
- `part_b_text_to_speech.py` — Part B, standalone: text → mp3 bytes.
  Test with: `python part_b_text_to_speech.py`
- `part_c_text_detection.py` — Part C, standalone: OCR on the image
  via `EasyOCR`, catches text BLIP can't read (signs, labels,
  documents) - a deep-learning scene-text reader, unlike Tesseract
  which is built for flat scanned documents and struggles on photos.
  Test with: `python part_c_text_detection.py path/to/image.jpg`
- `app.py` — Part D: full Streamlit app combining all three.
- `requirements.txt` — Python dependencies.

## Setup

Install the CPU-only build of PyTorch **first and separately** — the
default `pip install torch` pulls the CUDA/GPU build, which drags in
several huge Nvidia packages (100-200+ MB each: `nvidia-nccl`,
`nvidia-cusparselt`, etc.) that you don't need on CPU and that can
blow past disk quotas on shared machines:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Then install everything else:

```bash
pip install -r requirements.txt
```

EasyOCR downloads its own detection + recognition models (a few
hundred MB total) the first time it runs, then caches them locally —
no separate OS-level install needed. No API keys or GPU required
anywhere in the pipeline; BLIP-base and EasyOCR both run fine on CPU
(a few seconds per image).

## Run

```bash
streamlit run app.py --server.address=0.0.0.0 --server.port=8501
```

## Notes

- BLIP model loading is cached with `@st.cache_resource` and guarded by
  a `threading.Lock`, same as the Lab 7 app, so it's safe under
  Streamlit's rerun model.
- Output language defaults to English; switch to Nepali in the sidebar
  to get the same Nepali TTS behavior as Lab 7 Part D.
- `deep-translator`'s `GoogleTranslator` is used instead of
  `googletrans` (Part B in the lab) since it doesn't need `await` and
  is what the lab's own `app.py` uses.
- If you still hit disk-quota issues, clear pip's cache with
  `pip cache purge` before installing, and consider setting
  `TMPDIR`/`pip --cache-dir` to a location with more free space.