"""
Capstone Project: Accessibility Image Describer
Part A: Image Description (part_a_image_description.py)

Same pattern as Lab 7 Part A (BLIP captioning), tuned with beam
search + longer generation to get more detailed captions directly
from BLIP - no separate text-rewriting model needed (a small text-only
model can't add real detail since it never sees the image; it can
only paraphrase, which added no value). Runs entirely on CPU.

Test this file on its own first:
    python part_a_image_description.py path/to/image.jpg
"""

# Step 1: Import Required Libraries
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import sys

# Step 2: Load the Pretrained BLIP Processor and Model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")


def generate_caption(image: Image.Image) -> str:
    """
    Step 3-6: Preprocess -> Generate -> Decode, same as Lab 7 Part A,
    tuned for richer output:
      - num_beams: beam search instead of greedy decoding, picks a
        more globally coherent caption instead of the single most
        likely next word at each step.
      - max_new_tokens / min_length: pushes BLIP to produce a longer,
        more descriptive sentence instead of a 4-5 word fragment.
      - repetition_penalty: discourages the model from looping on the
        same phrase when generating longer output.
      - length_penalty > 1: rewards longer sequences during beam
        search (default penalizes them).
    """
    inputs = processor(image, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=60,
            min_length=15,
            num_beams=5,
            repetition_penalty=1.5,
            length_penalty=1.2,
        )
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption


def describe_image(image_path: str) -> str:
    """Full Part A pipeline: load -> caption."""
    image = Image.open(image_path).convert("RGB")
    return generate_caption(image)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python part_a_image_description.py <image_path>")
        sys.exit(1)

    caption = describe_image(sys.argv[1])
    print("Caption:", caption)