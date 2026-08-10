# فقط تست - هیچ import سنگینی نداره
import sys

# اول فایل لاگ رو باز کن
f = open('/kaggle/working/output.txt', 'w')

try:
    f.write("STEP 1: Starting\n")
    
    # تست ۱: nvidia-smi
    import subprocess
    r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10)
    f.write(f"GPU: {r.stdout}\n")
    
    # تست ۲: PyTorch
    f.write("STEP 2: Importing torch\n")
    import torch
    f.write(f"PyTorch: {torch.__version__}\n")
    f.write(f"CUDA available: {torch.cuda.is_available()}\n")
    
    if torch.cuda.is_available():
        f.write(f"GPU name: {torch.cuda.get_device_name(0)}\n")
        
        # یه محاسبه ساده CPU
        f.write("STEP 3: CPU test\n")
        a = torch.randn(100, 100)
        b = torch.randn(100, 100)
        c = a @ b
        f.write(f"CPU result: {c.sum().item()}\n")
        
        # یه محاسبه GPU
        f.write("STEP 4: GPU test\n")
        device = torch.device("cuda")
        x = torch.ones(1000, 1000, device=device)
        y = x + x
        f.write(f"GPU result: {y.sum().item()}\n")
    
    f.write("SUCCESS!\n")
    
except Exception as e:
    f.write(f"ERROR: {e}\n")
    import traceback
    f.write(traceback.format_exc())

f.close()
