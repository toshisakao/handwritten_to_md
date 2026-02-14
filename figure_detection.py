import re
import cv2
import torch
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from transformers import AutoProcessor, AutoModelForCausalLM

from utils import *

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

# YOLO DocLayNet
YOLO_REPO = "hantian/yolo-doclaynet"
YOLO_FILENAME = "yolov12m-doclaynet.pt"

# Florence-2 (replaces slow VLM calls for figure detection + verification)
FLORENCE_MODEL_ID = "microsoft/Florence-2-base"

# --- GLOBAL MODEL CACHE ---
yolo_model = None
florence_model = None
florence_processor = None

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_yolo():
    """Downloads and loads the YOLO Document Layout model."""
    global yolo_model
    print(f"⏳ Loading Detector (YOLO DocLayNet)...")

    try:
        model_path = hf_hub_download(repo_id=YOLO_REPO, filename=YOLO_FILENAME)
        yolo_model = YOLO(model_path)
        print(f"✅ YOLO Loaded! Classes: {yolo_model.names}")
    except Exception as e:
        print(f"❌ Failed to load YOLO: {e}")
        exit(1)


def load_florence():
    """Downloads and loads Florence-2 for figure detection & verification."""
    global florence_model, florence_processor
    print(f"⏳ Loading Florence-2 ({FLORENCE_MODEL_ID})...")

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        florence_model = AutoModelForCausalLM.from_pretrained(
            FLORENCE_MODEL_ID,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(device)

        florence_processor = AutoProcessor.from_pretrained(
            FLORENCE_MODEL_ID,
            trust_remote_code=True,
        )

        print(f"✅ Florence-2 Loaded on {device} ({dtype})")
    except Exception as e:
        print(f"❌ Failed to load Florence-2: {e}")
        exit(1)


def _florence_run(pil_image, task, text_input=""):
    """
    Run a Florence-2 inference task on a PIL image.
    Returns the parsed result dict.
    """
    device = florence_model.device
    dtype = florence_model.dtype

    prompt = task if not text_input else task + text_input

    inputs = florence_processor(
        text=prompt,
        images=pil_image,
        return_tensors="pt",
    ).to(device, dtype)

    with torch.no_grad():
        generated_ids = florence_model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            do_sample=False,
            num_beams=3,
        )

    generated_text = florence_processor.batch_decode(
        generated_ids, skip_special_tokens=False
    )[0]

    parsed = florence_processor.post_process_generation(
        generated_text, task=task, image_size=(pil_image.width, pil_image.height)
    )
    return parsed


# ---------------------------------------------------------------------------
# YOLO detection (pass 1 — fast)
# ---------------------------------------------------------------------------

def preprocess_for_yolo(pil_image):
    img_np = np.array(pil_image)
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return Image.fromarray(final_img)


def detect_figures_yolo(pil_image):
    """
    Uses YOLO DocLayNet to find figures/tables.
    """
    global yolo_model

    results = yolo_model(pil_image, conf=0.25, verbose=False)
    found_boxes = []
    result = results[0]

    for box in result.boxes:
        cls_id = int(box.cls[0])
        class_name = result.names[cls_id].lower()

        # DocLayNet classes: caption, footnote, formula, list-item,
        # page-footer, page-header, picture, section-header, table, text, title
        target_classes = ['picture', 'table']
        blocked_classes = ['formula', 'text', 'caption', 'footnote',
                           'page-footer', 'page-header', 'list-item',
                           'section-header', 'title']

        is_target = any(t in class_name for t in target_classes)
        is_blocked = any(b in class_name for b in blocked_classes)

        if is_target and not is_blocked:
            coords = box.xyxy[0].tolist()
            found_boxes.append(coords)

    return found_boxes


# ---------------------------------------------------------------------------
# Florence-2 detection (pass 2 — catches what YOLO misses)
# ---------------------------------------------------------------------------

def detect_figures_florence(pil_image):
    """
    Uses Florence-2 phrase grounding to locate figures on the page.
    Returns a list of [x1, y1, x2, y2] boxes in pixel coordinates.
    """
    task = "<CAPTION_TO_PHRASE_GROUNDING>"
    text_input = "diagram, graph, chart, drawing, sketch, figure, picture, table"

    try:
        result = _florence_run(pil_image, task, text_input)
        data = result.get(task, {})

        bboxes = data.get("bboxes", [])
        labels = data.get("labels", [])

        boxes = []
        for bbox, label in zip(bboxes, labels):
            if len(bbox) == 4:
                x1, y1, x2, y2 = [int(c) for c in bbox]
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
        return boxes

    except Exception as e:
        print(f"      ⚠️ Florence-2 detection failed: {e}")
        return []


def verify_figure_florence(pil_crop):
    """
    Uses Florence-2 captioning to determine if a crop is a figure/diagram
    or just text.  Returns True if it looks like a figure.
    """
    task = "<CAPTION>"

    try:
        result = _florence_run(pil_crop, task)
        caption = result.get(task, "").lower()

        # Figure-like keywords
        figure_kws = ['diagram', 'graph', 'chart', 'drawing', 'sketch',
                      'figure', 'picture', 'image', 'plot', 'illustration',
                      'table', 'circuit', 'map', 'flow', 'arrow', 'shape',
                      'circle', 'rectangle', 'line', 'curve', 'box']
        # Text-only keywords (reject if ONLY these match)
        text_kws = ['text', 'handwriting', 'writing', 'equation', 'formula',
                    'letter', 'word', 'sentence', 'paragraph', 'note',
                    'number', 'math']

        has_figure = any(kw in caption for kw in figure_kws)
        has_text = any(kw in caption for kw in text_kws)

        # Accept if it mentions figure-like content, even if also mentions text
        if has_figure:
            return True
        # Reject if it only mentions text-like content
        if has_text and not has_figure:
            return False
        # Ambiguous — keep it (err on inclusion)
        return True

    except Exception as e:
        print(f"      ⚠️ Florence-2 verification failed: {e}")
        return True


def refine_crop_florence(pil_crop):
    """
    Uses Florence-2 phrase grounding on a crop to find just the diagram
    region, excluding surrounding text.
    """
    task = "<CAPTION_TO_PHRASE_GROUNDING>"
    text_input = "diagram, graph, chart, drawing, figure"

    try:
        result = _florence_run(pil_crop, task, text_input)
        data = result.get(task, {})
        bboxes = data.get("bboxes", [])

        if not bboxes:
            return pil_crop

        # Pick the largest detected region
        best = max(bboxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        x1, y1, x2, y2 = [int(c) for c in best]

        # Sanity: refined box must be at least 25% of original
        refined_area = (x2 - x1) * (y2 - y1)
        original_area = pil_crop.width * pil_crop.height
        if refined_area > (original_area * 0.25) and x2 > x1 and y2 > y1:
            return pil_crop.crop((x1, y1, x2, y2))

    except Exception as e:
        print(f"      ⚠️ Florence-2 refinement failed: {e}")

    return pil_crop


# ---------------------------------------------------------------------------
# Box merging utilities
# ---------------------------------------------------------------------------

def _iou(box_a, box_b):
    """Compute intersection-over-union of two [x1,y1,x2,y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


def _overlap_ratio(box_a, box_b):
    """Fraction of box_a that overlaps with box_b."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    return inter / area_a if area_a > 0 else 0


def merge_boxes(yolo_boxes, florence_boxes, iou_threshold=0.3):
    """
    Merge two sets of bounding boxes, removing near-duplicates.
    When two boxes overlap significantly, keep the larger one.
    """
    all_boxes = list(yolo_boxes) + list(florence_boxes)
    if not all_boxes:
        return []

    all_boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)

    merged = []
    used = [False] * len(all_boxes)

    for i in range(len(all_boxes)):
        if used[i]:
            continue
        best = list(all_boxes[i])
        used[i] = True
        for j in range(i + 1, len(all_boxes)):
            if used[j]:
                continue
            if (_iou(best, all_boxes[j]) > iou_threshold
                    or _overlap_ratio(all_boxes[j], best) > 0.5):
                used[j] = True
        merged.append(best)

    return merged


# ---------------------------------------------------------------------------
# Hybrid detection pipeline (YOLO + Florence-2)
# ---------------------------------------------------------------------------

def detect_figures_hybrid(pil_image):
    """
    Two-pass hybrid figure detection:
      1. YOLO fast pass     — finds obvious 'picture' / 'table' regions
      2. Florence-2 scan    — phrase-grounded detection for missed figures
      3. Merge & dedupe     — combine both sets, remove overlaps
      4. Florence-2 verify  — caption-based false-positive filtering
    Returns a list of verified [x1, y1, x2, y2] boxes.
    """
    width, height = pil_image.size
    page_area = width * height

    # --- Pass 1: YOLO ---
    yolo_boxes = detect_figures_yolo(pil_image)
    print(f"      YOLO found {len(yolo_boxes)} candidate(s)")

    # --- Pass 2: Florence-2 ---
    florence_boxes = detect_figures_florence(pil_image)
    print(f"      Florence-2 found {len(florence_boxes)} candidate(s)")

    # --- Merge ---
    merged = merge_boxes(yolo_boxes, florence_boxes)
    print(f"      Merged to {len(merged)} unique region(s)")

    # --- Filter obvious bad boxes ---
    filtered = []
    for box in merged:
        x1, y1, x2, y2 = map(int, box)
        box_area = (x2 - x1) * (y2 - y1)
        if box_area > (page_area * 0.50):
            continue
        if (x2 - x1) < 50 or (y2 - y1) < 50:
            continue
        filtered.append([x1, y1, x2, y2])

    # --- Pass 3: Florence-2 verification ---
    verified = []
    for box in filtered:
        x1, y1, x2, y2 = box
        pad = 10
        cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
        cx2, cy2 = min(width, x2 + pad), min(height, y2 + pad)
        crop = pil_image.crop((cx1, cy1, cx2, cy2))

        if verify_figure_florence(crop):
            verified.append(box)
            print(f"      ✓ Region ({x1},{y1})-({x2},{y2}) verified as figure")
        else:
            print(f"      ✗ Region ({x1},{y1})-({x2},{y2}) rejected (text/equation)")

    print(f"      Final: {len(verified)} verified figure(s)")
    return verified


# ---------------------------------------------------------------------------
# Crop refinement & saving
# ---------------------------------------------------------------------------

def smart_crop_whitespace(pil_image):
    """
    Splits image based on vertical whitespace gaps (columns).
    Keeps the largest segment (the graph).
    """
    img = np.array(pil_image.convert('L'))
    img = 255 - img
    _, img = cv2.threshold(img, 20, 1, cv2.THRESH_BINARY)

    projection = np.sum(img, axis=0)
    height = img.shape[0]
    is_ink = projection > (height * 0.01)

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

    if not segments:
        return pil_image

    best_segment = max(segments, key=lambda s: np.sum(projection[s[0]:s[1]]))
    x1, x2 = best_segment
    x1 = max(0, x1 - 10)
    x2 = min(pil_image.width, x2 + 10)

    return pil_image.crop((x1, 0, x2, pil_image.height))


def is_crop_dirty(pil_crop):
    """
    Runs YOLO on the crop to check for significant text blocks.
    Only marks dirty if text covers >10% of the crop area.
    """
    results = yolo_model(pil_crop, conf=0.3, verbose=False)
    result = results[0]
    crop_area = pil_crop.width * pil_crop.height

    for box in result.boxes:
        cls_id = int(box.cls[0])
        class_name = result.names[cls_id].lower()

        if class_name in ['text', 'caption', 'list-item']:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            text_area = (x2 - x1) * (y2 - y1)
            if text_area > (crop_area * 0.10):
                return True
    return False


def crop_and_save_figures(original_img, bboxes, page_num, pdf_name, figures_dir):
    """
    Crops detected figure regions, refines them, and saves to disk.
    Accepts pre-filtered bboxes from the hybrid detection pipeline.
    Uses Florence-2 for refinement (no Ollama VLM needed).
    """
    saved_md_links = []
    width, height = original_img.size
    figures_dir.mkdir(parents=True, exist_ok=True)

    for idx, box in enumerate(bboxes):
        x1, y1, x2, y2 = map(int, box)

        # Padding
        pad = 20
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(width, x2 + pad), min(height, y2 + pad)

        # --- CROP & REFINE ---
        crop = original_img.crop((x1, y1, x2, y2))

        # Whitespace cleaning pass
        crop = smart_crop_whitespace(crop)

        # Loop whitespace if still dirty (limit to 3 passes)
        dirty_counter = 0
        while is_crop_dirty(crop) and dirty_counter < 3:
            crop = smart_crop_whitespace(crop)
            dirty_counter += 1

        # Final Florence-2 refinement if still dirty
        if is_crop_dirty(crop):
            crop = refine_crop_florence(crop)

        # Save
        clean_name = pdf_name.replace(".pdf", "").replace(" ", "_")
        filename = f"{clean_name}_p{page_num}_fig{idx+1}.jpg"
        save_path = figures_dir / filename
        crop.save(save_path, quality=95)

        rel_path = f"figures/{filename}"
        saved_md_links.append(f"![Figure {idx+1}]({rel_path})")

    return saved_md_links
