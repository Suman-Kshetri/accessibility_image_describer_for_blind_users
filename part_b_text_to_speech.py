"""
Capstone Project: Accessibility Image Describer
Part B: Text-to-Speech (part_b_text_to_speech.py)

Same pattern as Lab 7 Part C (tts.py), wrapped as a reusable function
so app.py can call it directly without writing a temp file.

Test this file on its own first:
    python part_b_text_to_speech.py
"""

# Step 1: Import Required Libraries
from gtts import gTTS
from io import BytesIO


def text_to_speech_bytes(text: str, lang: str = "en") -> bytes:
    """
    Convert text to speech and return raw mp3 bytes (no temp file),
    the same approach used in app.py of Lab 7 Part D.
    """
    tts = gTTS(text=text, lang=lang)
    mp3_bytes = BytesIO()
    tts.write_to_fp(mp3_bytes)
    mp3_bytes.seek(0)
    return mp3_bytes.read()


if __name__ == "__main__":
    # Quick standalone test, same shape as the lab's tts.py
    sample_text = "A dog is running across a green field."
    audio_bytes = text_to_speech_bytes(sample_text, lang="en")

    with open("test_speech.mp3", "wb") as f:
        f.write(audio_bytes)

    print("Saved test_speech.mp3 -", len(audio_bytes), "bytes")