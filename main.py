import os
import io
import shutil
import subprocess
import cv2
import numpy as np
import ollama
import torch
import warnings
from pathlib import Path
from pdf2image import convert_from_path
from tqdm import tqdm
from PIL import Image
from huggingface_hub import hf_hub_download
from ultralytics import YOLO 

# --- CONFIGURATION ---
input_path = "./test/"
output_path = "./test_out"

# READER MODEL (Ollama)
OCR_MODEL = "qwen3-vl"

# DETECTOR MODEL (YOLOv8 DocLayNet)
# This model is specifically trained to find 'Pictures', 'Tables', 'Formulas' in docs.
YOLO_REPO = "hantian/yolo-doclaynet"
YOLO_FILENAME = "yolov8n-doclaynet.pt"

DPI_SETTING = 300

# --- SILENCE WARNINGS ---
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# --- GLOBAL MODEL CACHE ---
yolo_model = None

def check_gpu_status():
    if shutil.which("nvidia-smi"):
        subprocess.run(['nvidia-smi', '--query-gpu=name,memory.used,memory.total', '--format=csv,noheader'])

def load_yolo():
    """Downloads and loads the YOLOv8 Document Layout model."""
    global yolo_model
    print(f"⏳ Loading Detector (YOLOv8 DocLayNet)...")
    
    try:
        # 1. Download weights from HuggingFace
        model_path = hf_hub_download(repo_id=YOLO_REPO, filename=YOLO_FILENAME)
        
        # 2. Load model using standard Ultralytics syntax
        yolo_model = YOLO(model_path)
        
        print(f"✅ Detector Loaded! Classes: {yolo_model.names}")
        
    except Exception as e:
        print(f"❌ Failed to load YOLO: {e}")
        print("   (Check your internet connection or HuggingFace status)")
        exit(1)

def pil_to_bytes(pil_image):
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format='JPEG', quality=95)
    return img_byte_arr.getvalue()

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

def detect_figures_yolo(pil_image):
    """
    Uses YOLOv8 to find figures/tables using standard Ultralytics API.
    """
    global yolo_model

    # 1. Perform object detection on an image
    #    conf=0.25 is standard confidence threshold
    results = yolo_model(pil_image, conf=0.25, verbose=False)
    
    found_boxes = []
    
    # 2. Extract results
    #    The result is a list (one per image), we only processed one.
    result = results[0]
    
    # 3. Iterate through detected boxes
    for box in result.boxes:
        # Get the class ID (int)
        cls_id = int(box.cls[0])
        # Get the class name (string)
        class_name = result.names[cls_id].lower()
        
        # --- STRICT FILTERING ---
        # We explicitly WANT these:
        target_classes = ['picture', 'image', 'chart', 'graph', 'drawing']
        
        # We explicitly BLOCK these:
        blocked_classes = ['formula', 'math', 'equation', 'text', 'caption', 'footer', 'header']
        
        # Check if it matches a target AND is not blocked
        is_target = any(t in class_name for t in target_classes)
        is_blocked = any(b in class_name for b in blocked_classes)

        if is_target and not is_blocked:
            # Get coordinates [x1, y1, x2, y2]
            coords = box.xyxy[0].tolist()
            found_boxes.append(coords)
                
    return found_boxes

def crop_and_save_figures(original_img, bboxes, page_num, pdf_name, figures_dir):
    saved_md_links = []
    width, height = original_img.size
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate Page Area for safety filter
    page_area = width * height

    for idx, box in enumerate(bboxes):
        x1, y1, x2, y2 = map(int, box)

        # 1. Whole Page Filter
        # If a single box covers > 60% of the page, it's likely a false positive (e.g. background)
        box_area = (x2 - x1) * (y2 - y1)
        if box_area > (page_area * 0.60):
            continue

        # 2. Padding (Add 20px border so we don't cut lines)
        pad = 20
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(width, x2 + pad)
        y2 = min(height, y2 + pad)

        # 3. Tiny Box Filter
        if (x2 - x1) < 50 or (y2 - y1) < 50:
            continue

        crop = original_img.crop((x1, y1, x2, y2))
        
        clean_name = pdf_name.replace(".pdf", "").replace(" ", "_")
        filename = f"{clean_name}_p{page_num}_fig{idx+1}.jpg"
        save_path = figures_dir / filename
        crop.save(save_path, quality=95)
        
        rel_path = f"figures/{filename}"
        saved_md_links.append(f"![Figure {idx+1}]({rel_path})")
        
    return saved_md_links

def transcribe_text(image_bytes):
    SYSTEM_PROMPT = """
    Transcribe the handwritten text in this image into Markdown.
    1. Ignore diagrams (we already extracted them).
    2. Use # for headers and - for lists.
    3. Use LaTeX ($) for math.
    """
    try:
        response = ollama.chat(
            model=OCR_MODEL,
            messages=[{'role': 'user', 'content': SYSTEM_PROMPT, 'images': [image_bytes]}]
        )
        return response['message']['content']
    except Exception as e:
        return f"> **OCR Error:** {e}"

def process_pdf(pdf_path, output_file):
    try:
        images = convert_from_path(pdf_path, dpi=DPI_SETTING)
    except Exception as e:
        print(f"❌ Read Error {pdf_path.name}: {e}")
        return

    full_text = f"# {pdf_path.name}\n\n"
    figures_dir = output_file.parent / "figures"
    
    # Progress Bar (Pages)
    pbar = tqdm(enumerate(images), total=len(images), desc="Init", leave=False)
    
    for i, original_img in pbar:
        # 1. Detection (YOLO)
        pbar.set_description(f"Page {i+1} | Detecting (YOLOv8)")
        bboxes = detect_figures_yolo(original_img)
        
        # 2. Cropping
        pbar.set_description(f"Page {i+1} | Cropping {len(bboxes)} figs")
        fig_links = crop_and_save_figures(original_img, bboxes, i+1, pdf_path.name, figures_dir)
        
        user_input = input(f"Continue with preprocessing? [Y/n]:").strip().lower()
        if user_input == 'n':
            print(f" * Skipping!")
            continue 

        # 3. Preprocessing
        pbar.set_description(f"Page {i+1} | Preprocessing")
        processed_img = preprocess_for_ocr(original_img)
        processed_bytes = pil_to_bytes(processed_img)

        # 4. OCR (Qwen3)
        pbar.set_description(f"Page {i+1} | Reading ({OCR_MODEL})")
        text_content = transcribe_text(processed_bytes)
        
        # 5. Save
        full_text += f"## Page {i + 1}\n\n"
        if fig_links:
            full_text += "**Figures:**\n\n" + "\n\n".join(fig_links) + "\n\n---\n\n"
        full_text += f"{text_content}\n\n---\n\n"

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(full_text)

def main():
    check_gpu_status()
    load_yolo() 
    
    in_dir = Path(input_path)
    out_dir = Path(output_path)

    pdf_files = list(in_dir.rglob("*.pdf"))
    print(f"📂 Found {len(pdf_files)} PDFs.")

    file_pbar = tqdm(pdf_files, unit="file", desc="Total Progress")
    for pdf in file_pbar:
        rel = pdf.relative_to(in_dir)
        out_file = out_dir / rel.with_suffix(".md")
        
        file_pbar.set_description(f"Processing: {pdf.name}")

        if out_file.exists():
            # Clean Overwrite Prompt
            file_pbar.clear()
            print(f"\n⚠️  File exists: {out_file.name}")
            user_input = input(f"   Overwrite? [Y/n]: ").strip().lower()
            file_pbar.refresh()

            if user_input == 'n':
                file_pbar.write(f"⏩ Skipping {pdf.name}")
                continue
            else:
                file_pbar.write(f"♻️  Overwriting {pdf.name}...")
        
        process_pdf(pdf, out_file)

if __name__ == "__main__":
    main()
