import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration, logging
from PIL import Image
import torch
import numpy as np
from gtts import gTTS
from io import BytesIO
import threading
import os
import easyocr

# Suppress transformers warnings (same as Lab 7 Part D)
logging.set_verbosity_error()

# Use all available CPU cores for BLIP's beam search / conv ops
# instead of PyTorch's conservative default thread count.
torch.set_num_threads(os.cpu_count())

# Lock for thread-safe model use (same as Lab 7 Part D)
model_lock = threading.Lock()

# Downscale very large camera/upload images before running BLIP/OCR.
# Neither model needs full sensor resolution, and OCR in particular
# (running unquantized, see load_ocr_reader) gets much slower on
# large images since detection + recognition both scale with pixel
# count.
MAX_IMAGE_DIM = 1024


# ---------------------------------------------------------------------
# Model loading (cached, same pattern as Lab 7)
# ---------------------------------------------------------------------
@st.cache_resource
def load_caption_model():
    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base",
        local_files_only=False,
    )
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base",
        local_files_only=False,
    )
    return processor, model


@st.cache_resource
def load_ocr_reader():
    # EasyOCR: deep-learning scene-text reader. Handles text embedded
    # in photos (signs, labels) much better than Tesseract, which is
    # built for flat scanned documents.
    # quantize=False: EasyOCR's default dynamic quantization routes
    # through PyTorch's FBGEMM backend, which requires AVX2/AVX512-VNNI.
    # On CPUs without those instructions this causes a silent
    # "Illegal instruction (core dumped)" crash mid-inference rather
    # than a catchable Python exception, so it's disabled here.
    return easyocr.Reader(["en"], gpu=False, verbose=False, quantize=False)


processor, model = load_caption_model()
ocr_reader = load_ocr_reader()


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def resize_for_inference(image: Image.Image) -> Image.Image:
    """Cap the longest side at MAX_IMAGE_DIM, keeping aspect ratio."""
    w, h = image.size
    longest = max(w, h)
    if longest <= MAX_IMAGE_DIM:
        return image
    scale = MAX_IMAGE_DIM / longest
    return image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


@st.cache_data(show_spinner=False)
def generate_caption_cached(image_bytes: bytes) -> str:
    """
    Cached by raw image bytes so re-running the Streamlit script
    (e.g. toggling autoplay) doesn't re-run BLIP on an image it's
    already captioned.

    Tuned for richer output (see part_a_image_description.py for the
    rationale behind each generation parameter):
      - beam search (num_beams=3) instead of greedy decoding
      - longer minimum/maximum length so BLIP produces a fuller
        sentence, not a 4-5 word fragment
      - repetition_penalty + length_penalty to keep longer captions
        coherent instead of looping
    num_beams was dropped from 5 to 3: on CPU, beam search cost
    scales roughly linearly with beam count, so this alone cuts
    captioning time by ~40% with only a marginal drop in caption
    detail. Bump it back to 5 if quality matters more than latency.
    """
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image = resize_for_inference(image)
    with model_lock:
        inputs = processor(image, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=50,
                min_length=15,
                num_beams=3,
                repetition_penalty=1.5,
                length_penalty=1.2,
            )
        return processor.decode(out[0], skip_special_tokens=True)


@st.cache_data(show_spinner=False)
def text_to_speech_bytes_cached(text: str, lang: str = "en") -> bytes:
    """Cached by (text, lang) so re-runs don't hit gTTS again for the
    same spoken text, e.g. when the user only flips the autoplay
    checkbox after audio has already been generated."""
    tts = gTTS(text=text, lang=lang)
    mp3_bytes = BytesIO()
    tts.write_to_fp(mp3_bytes)
    mp3_bytes.seek(0)
    return mp3_bytes.read()


OCR_CONFIDENCE_THRESHOLD = 0.2


@st.cache_data(show_spinner=False)
def extract_text_cached(image_bytes: bytes) -> str:
    """
    OCR pass to catch text BLIP can't read (signs, labels, documents).
    Drops low-confidence detections (usually noise from branches,
    edges, etc.) rather than reading them aloud. Returns "" if
    nothing confident is found. Cached by image bytes for the same
    reason as generate_caption_cached above.
    """
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image = resize_for_inference(image)
    results = ocr_reader.readtext(np.array(image))
    words = [text for (_, text, conf) in results if conf >= OCR_CONFIDENCE_THRESHOLD]
    return " ".join(words)


# ---------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------
st.title("🦯 Accessibility Image Describer")
st.caption(
    "Capstone project: image → detailed "
    "caption → speech, for blind/low-vision users."
)

with st.sidebar:
    st.header("Settings")
    input_mode = st.radio(
        "Image source",
        options=["Upload", "Live Capture"],
    )
    detect_text = st.checkbox("Read text in image (OCR)", value=True)
    autoplay = st.checkbox("Autoplay audio", value=True)

image_file = None
if input_mode == "Upload":
    image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
else:
    image_file = st.camera_input("Capture from camera")

if image_file:
    image_bytes = image_file.getvalue()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    st.image(image, caption="Selected Image", use_container_width=True)

    with st.spinner("Looking at the image..."):
        caption = generate_caption_cached(image_bytes)

    st.subheader("Description")
    st.write(caption)

    spoken_text = caption

    if detect_text:
        with st.spinner("Checking for text in the image..."):
            detected_text = extract_text_cached(image_bytes)
        if detected_text:
            st.subheader("Text Found in Image")
            st.write(detected_text)
            spoken_text = f"{caption}. The street sign in the image reads: {detected_text}"

    with st.spinner("Generating audio..."):
        audio_bytes = text_to_speech_bytes_cached(spoken_text, lang="en")

    st.subheader("Listen")
    st.audio(audio_bytes, format="audio/mp3", autoplay=autoplay)
else:
    st.info("Upload an image or use your camera to get a spoken description.")