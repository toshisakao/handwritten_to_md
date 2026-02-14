import io
import shutil
import subprocess

def check_gpu_status():
    if shutil.which("nvidia-smi"):
        subprocess.run(['nvidia-smi', '--query-gpu=name,memory.used,memory.total', '--format=csv,noheader'])

def pil_to_bytes(pil_image):
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format='JPEG', quality=95)
    return img_byte_arr.getvalue()

