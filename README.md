# handwritten_to_md

Converts handwritten PDF notes into clean Markdown, with automatic figure/diagram extraction.

Python 3.12

## Dependencies

### System

- **poppler** - required by `pdf2image` for PDF rendering
  - Ubuntu/Debian: `sudo apt install poppler-utils`
  - Arch: `sudo pacman -S poppler`
  - macOS: `brew install poppler`
- **Ollama** - local LLM runtime for OCR (text transcription)
  - https://ollama.com
  - Pull the OCR model: `ollama pull qwen3-vl`
- **NVIDIA GPU** with CUDA support (tested on RTX 3070 Ti 8GB)

### Python

Install all Python dependencies:

```bash
pip install -r requirements.txt
```

Key packages:

| Package | Purpose |
|---------|---------|
| `torch`, `torchvision` | PyTorch backend (CUDA) |
| `transformers` | Florence-2 model for figure detection & verification |
| `timm`, `einops` | Required by Florence-2 architecture |
| `ultralytics` | YOLOv12 DocLayNet for layout detection |
| `huggingface_hub` | Model downloading (YOLO weights, Florence-2) |
| `ollama` | Qwen3-VL for handwritten text OCR |
| `opencv-python` | Image preprocessing (CLAHE, thresholding, denoising) |
| `pdf2image` | PDF page to image conversion |
| `pillow` | Image handling |
| `tqdm` | Progress bars |
| `numpy` | Array operations |

### Models (downloaded automatically on first run)

| Model | Source | Size | Purpose |
|-------|--------|------|---------|
| YOLOv12m-DocLayNet | `hantian/yolo-doclaynet` | ~50MB | Fast layout detection (pass 1) |
| Florence-2-base | `microsoft/Florence-2-base` | ~0.5GB | Figure detection, verification & crop refinement (pass 2) |
| Qwen3-VL | Ollama (`qwen3-vl`) | ~5GB | Handwritten text OCR |

## Usage

1. Place PDF files in `./test/`
2. Run:

```bash
python main.py
```

3. Output Markdown and extracted figures are saved to `./test_out/`
