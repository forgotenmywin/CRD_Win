print("======================================")
print("        USER JOB TEST")
print("======================================")

import torch

print("CUDA:", torch.cuda.is_available())

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

print("USER JOB IS RUNNING")

x = torch.randn(
    4096,
    4096,
    device="cuda"
)

y = torch.randn(
    4096,
    4096,
    device="cuda"
)

z = x @ y

torch.cuda.synchronize()

print("Calculation completed.")
print("Result shape:", z.shape)
