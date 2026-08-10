# این کد روی Kaggle GPU اجرا می‌شه
import torch
import subprocess

# GPU Info
smi = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
print("GPU:", smi.stdout)

# محاسبه
print("Starting...")
device = torch.device("cuda")
a = torch.randn(5000, 5000, device=device)
b = torch.randn(5000, 5000, device=device)

torch.cuda.synchronize()
c = torch.matmul(a, b)
torch.cuda.synchronize()

print("Result:", c.shape)
print("Sum:", c.sum().item())
print("Done!")

