import ollama
import cv2
import numpy as np
from PIL import Image



def preprocess_for_ocr(pil_image):
    """High-contrast preprocessing for TEXT reading."""
    img_np = np.array(pil_image)
    if len(img_np.shape) == 2:
        gray = img_np
    else:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
    return Image.fromarray(denoised)


def transcribe_text(image_bytes, model):
    SYSTEM_PROMPT = """
    You are an advanced OCR engine specialized in handwritten text. 
    Transcribe the content of this image into clean Markdown. 
    1. Transcribe text exactly as written.
    2. Use Markdown headers (#) for big text and bullet points (-) for lists.
    3. Use latex math ($) for math notations. 
    4. Do not add conversational filler like "Here is the text". Just give the markdown.
    """
    try:
        response = ollama.chat(
            model=model,
            messages=[{'role': 'user', 'content': SYSTEM_PROMPT, 'images': [image_bytes]}]
        )
        return response['message']['content']
    except Exception as e:
        return f"> **OCR Error:** {e}"


