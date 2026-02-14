import re
import cv2
import ollama
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
from ultralytics import YOLO 

from utils import *

# DETECTOR MODEL (YOLOv8 DocLayNet)
# This model is specifically trained to find 'Pictures', 'Tables', 'Formulas' in docs.
# YOLO_REPO = "hantian/yolo-doclaynet"
# YOLO_FILENAME = "yolov8n-doclaynet.pt"

YOLO_REPO = "hantian/yolo-doclaynet"
YOLO_FILENAME = "yolov12m-doclaynet.pt"  # Medium is significantly better than Nano

# --- GLOBAL MODEL CACHE ---
yolo_model = None

def load_yolo():
    """Downloads and loads the YOLO Document Layout model."""
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

def preprocess_for_yolo(pil_image):
    img_np = np.array(pil_image)
    # Convert to LAB color space to process lightness only
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to the L-channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    
    # Merge back and convert to RGB
    limg = cv2.merge((cl, a, b))
    final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return Image.fromarray(final_img)

def detect_figures_yolo(pil_image):
    """
    Uses YOLOv8 to find figures/tables using standard Ultralytics API.
    """
    global yolo_model

    # 1. Perform object detection on an image
    #    conf=0.25 is standard confidence threshold
    results = yolo_model(pil_image, conf=0.2, verbose=False)
    
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

def smart_crop_whitespace(pil_image):
    """
    Splits image based on vertical whitespace gaps (columns).
    Keeps the largest segment (the graph).
    """
    # 1. Convert to binary numpy array (0 = white, 1 = ink)
    img = np.array(pil_image.convert('L'))
    # Invert: Text is now high value (255), paper is 0
    img = 255 - img
    # Threshold to pure 0/1
    _, img = cv2.threshold(img, 20, 1, cv2.THRESH_BINARY)

    # 2. Calculate Vertical Projection (Sum of ink in each column)
    # shape: (width,)
    projection = np.sum(img, axis=0)

    # 3. Find gaps (columns with very little ink)
    height = img.shape[0]
    # If a column has less than 1% ink coverage, it's a gap
    is_ink = projection > (height * 0.01) 
    
    # 4. Find connected segments of "Ink"
    segments = []
    start = None
    for x, has_ink in enumerate(is_ink):
        if has_ink and start is None:
            start = x
        elif not has_ink and start is not None:
            segments.append((start, x))
            start = None
    if start is not None:
        segments.append((start, len(is_ink)))

    # 5. If no separation found, return original
    if not segments:
        return pil_image

    # 6. Find the "heaviest" segment (most ink total)
    # This assumes the graph has more ink density/area than the text note
    best_segment = max(segments, key=lambda s: np.sum(projection[s[0]:s[1]]))
    
    # 7. Crop to that segment
    x1, x2 = best_segment
    # Add a little padding (10px)
    x1 = max(0, x1 - 10)
    x2 = min(pil_image.width, x2 + 10)
    
    return pil_image.crop((x1, 0, x2, pil_image.height))

def is_crop_dirty(pil_crop):
    """
    Runs YOLO on the crop to see if it contains significant text blocks.
    """
    # Run YOLO with a slightly higher confidence to avoid false positives
    results = yolo_model(pil_crop, conf=0.3, verbose=False)
    result = results[0]
    
    for box in result.boxes:
        cls_id = int(box.cls[0])
        class_name = result.names[cls_id].lower()
        
        # If YOLO finds 'text' or 'caption' inside our figure crop, it's dirty
        if class_name in ['text', 'caption', 'list-item']:
            # Optional: Only mark dirty if the text box is large
            # x1, y1, x2, y2 = box.xyxy[0].tolist()
            # if (x2-x1)*(y2-y1) > (pil_crop.width * pil_crop.height * 0.1):
            return True
    return False

def refine_crop_with_vlm(pil_crop, model):
    """
    Sends a 'dirty' crop to Qwen3-VL and asks it to find the 
    precise coordinates of JUST the diagram, ignoring the text.
    """
    img_bytes = pil_to_bytes(pil_crop)
    
    # We use a very strict prompt for visual grounding
    PROMPT = """
    In this image, there is a diagram mixed with handwritten text. 
    Find the bounding box of ONLY the diagram/graph. 
    Exclude all surrounding math equations and text.
    Return the coordinates in JSON format: {"box": [ymin, xmin, ymax, xmax]} (scale 0-1000).
    """

    try:
        response = ollama.chat(
            model,
            messages=[{'role': 'user', 'content': PROMPT, 'images': [img_bytes]}]
        )
        content = response['message']['content']
        
        # Look for the coordinate pattern [ymin, xmin, ymax, xmax]
        match = re.search(r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]', content)
        if match:
            # Parse coordinates (Qwen uses 0-1000 normalized scale)
            ymin, xmin, ymax, xmax = map(int, match.groups())
            
            w, h = pil_crop.size
            left = xmin * w / 1000
            top = ymin * h / 1000
            right = xmax * w / 1000
            bottom = ymax * h / 1000
            
            # Perform the sub-crop
            return pil_crop.crop((left, top, right, bottom))
    except Exception as e:
        print(f"      ⚠️ VLM Refinement failed: {e}")
    
    return pil_crop # Fallback to original if AI fails

def crop_and_save_figures(original_img, bboxes, page_num, pdf_name, figures_dir, model):
    saved_md_links = []
    width, height = original_img.size
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    page_area = width * height

    for idx, box in enumerate(bboxes):
        x1, y1, x2, y2 = map(int, box)

        # 1. Whole Page Filter (Adjusted to 0.40 as you requested)
        box_area = (x2 - x1) * (y2 - y1)
        if box_area > (page_area * 0.40):
            continue

        # 2. Padding
        pad = 20
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(width, x2 + pad), min(height, y2 + pad)

        # 3. Tiny Box Filter
        if (x2 - x1) < 50 or (y2 - y1) < 50:
            continue

        # --- CROP & REFINE ---
        crop = original_img.crop((x1, y1, x2, y2))
        
        # First attempt: Whitespace cleaning
        crop = smart_crop_whitespace(crop)
        
        # Second attempt: Loop whitespace if still dirty
        dirty_counter = 0
        while is_crop_dirty(crop) and dirty_counter < 10: # Reduced to 3 for speed
            crop = smart_crop_whitespace(crop)
            dirty_counter += 1
        
        # --- NEW: FINAL VLM BEGGING PHASE ---
        crop = refine_crop_with_vlm(crop, model)
        
        # Save logic
        clean_name = pdf_name.replace(".pdf", "").replace(" ", "_")
        filename = f"{clean_name}_p{page_num}_fig{idx+1}.jpg"
        save_path = figures_dir / filename
        crop.save(save_path, quality=95)
        
        rel_path = f"figures/{filename}"
        saved_md_links.append(f"![Figure {idx+1}]({rel_path})")
        
    return saved_md_links


