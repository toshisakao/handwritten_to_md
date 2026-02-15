import os
import warnings
from pathlib import Path
from pdf2image import convert_from_path
from tqdm import tqdm

from utils import *
from figure_detection import *
from ocr import preprocess_for_ocr, transcribe_text

# --- CONFIGURATION ---
input_path = "./test/"
output_path = "./test_out"

# READER MODEL (Ollama)
OCR_MODEL = "qwen3-vl"


DPI_SETTING = 300

# --- SILENCE WARNINGS ---
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

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
        # 0. preprocess into yolo
        # detect_processed_img = preprocess_for_yolo(original_img)

        # 1. Hybrid Detection (YOLO + Florence-2)
        pbar.set_description(f"Page {i+1} | Detecting (Hybrid)")
        bboxes = detect_figures_qwen(original_img, OCR_MODEL)

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
        text_content = transcribe_text(processed_bytes, OCR_MODEL)
        
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
    load_florence()
    
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
