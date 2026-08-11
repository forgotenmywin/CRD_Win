import torch
import time

print("======================================")
print("        GPU WORKER JOB")
print("======================================")

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is not available")

print("GPU:", torch.cuda.get_device_name(0))
print("CUDA:", torch.version.cuda)

# GPU test
size = 4096

print(f"\nRunning GPU calculation: {size}x{size}")

a = torch.randn(size, size, device="cuda")
b = torch.randn(size, size, device="cuda")

torch.cuda.synchronize()

start = time.perf_counter()

c = torch.matmul(a, b)

torch.cuda.synchronize()

elapsed = time.perf_counter() - start

print("\n======================================")
print("GPU JOB FINISHED")
print("======================================")

print("GPU:", torch.cuda.get_device_name(0))
print("Result:", c.shape)
print(f"Time: {elapsed:.4f} seconds")
