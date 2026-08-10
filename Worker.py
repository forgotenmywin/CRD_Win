import subprocess
import sys


def run(command):
    print(f"\n>>> {command}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        sys.exit(result.returncode)


print("================================")
print("     Kaggle GPU Worker Test")
print("================================")

print("\n[1] Checking Python...")
print(sys.version)

print("\n[2] Checking NVIDIA GPU...")
run("nvidia-smi")

print("\n[3] Checking CUDA...")
try:
    import torch

    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA version:", torch.version.cuda)

        x = torch.randn(4096, 4096, device="cuda")
        y = torch.randn(4096, 4096, device="cuda")

        torch.cuda.synchronize()

        z = x @ y

        torch.cuda.synchronize()

        print("GPU computation: OK")
        print("Result shape:", z.shape)

    else:
        print("CUDA GPU is not available")

except Exception as e:
    print("PyTorch error:", e)

print("\n================================")
print("Worker finished")
print("================================")
