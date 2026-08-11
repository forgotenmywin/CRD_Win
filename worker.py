import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


USERNAME = os.environ["KAGGLE_USERNAME"]
KERNEL = f"{USERNAME}/notebookd9d7092a0a"

WORK_DIR = Path("kaggle_kernel")
OUTPUT_DIR = Path("kaggle_output")


def run(command, check=True):
    print(f"\n>>> {command}", flush=True)

    result = subprocess.run(
        command,
        shell=True,
        text=True
    )

    if check and result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    return result.returncode


print("======================================", flush=True)
print("       KAGGLE GPU GENERIC WORKER", flush=True)
print("======================================", flush=True)


# ==================================================
# 1. Check credentials
# ==================================================

if not os.getenv("KAGGLE_USERNAME"):
    sys.exit("KAGGLE_USERNAME is missing")

if not os.getenv("KAGGLE_API_TOKEN"):
    sys.exit("KAGGLE_API_TOKEN is missing")

print("[OK] Kaggle credentials found", flush=True)


# ==================================================
# 2. Clean directories
# ==================================================

if WORK_DIR.exists():
    shutil.rmtree(WORK_DIR)

if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)

WORK_DIR.mkdir(parents=True)
OUTPUT_DIR.mkdir(parents=True)


# ==================================================
# 3. Pull Kaggle Notebook
# ==================================================

print("\n[1] Pulling Kaggle Notebook...", flush=True)

run(
    f"python -m kaggle kernels pull "
    f"{KERNEL} "
    f"-p {WORK_DIR} "
    f"-m"
)


# ==================================================
# 4. Find notebook
# ==================================================

notebook_file = next(
    WORK_DIR.glob("*.ipynb"),
    None
)

if notebook_file is None:
    sys.exit("Notebook .ipynb file not found")

print(
    f"[OK] Notebook found: {notebook_file}",
    flush=True
)


# ==================================================
# 5. Read job.py
# ==================================================

job_file = Path("job.py")

if not job_file.exists():
    sys.exit("job.py not found")

job_code = job_file.read_text(
    encoding="utf-8"
)

print("[OK] job.py loaded", flush=True)


# ==================================================
# 6. Load notebook
# ==================================================

notebook = json.loads(
    notebook_file.read_text(
        encoding="utf-8"
    )
)


# ==================================================
# 7. Create worker code
# ==================================================

runner_code = r'''
print("======================================")
print("       KAGGLE GPU WORKER START")
print("======================================")

import sys

print("Python:", sys.version)

try:
    import torch

    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is NOT available")

    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)

    print("GPU test starting...")

    import time

    size = 2048

    a = torch.randn(
        size,
        size,
        device="cuda"
    )

    b = torch.randn(
        size,
        size,
        device="cuda"
    )

    torch.cuda.synchronize()

    start = time.perf_counter()

    c = torch.matmul(a, b)

    torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    print("======================================")
    print("GPU TEST SUCCESS")
    print("======================================")

    print("GPU:", torch.cuda.get_device_name(0))
    print("Result shape:", c.shape)
    print("Time:", elapsed)

except Exception as e:

    print("======================================")
    print("GPU TEST ERROR")
    print("======================================")

    print(type(e).__name__)
    print(str(e))

    raise


print("======================================")
print("       RUNNING USER JOB")
print("======================================")

''' + job_code + r'''

print("======================================")
print("       USER JOB FINISHED")
print("======================================")
'''


# ==================================================
# 8. Replace notebook cells
# ==================================================

notebook["cells"] = [
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": runner_code.splitlines(True)
    }
]


# ==================================================
# 9. Configure GPU
# ==================================================

metadata_file = WORK_DIR / "kernel-metadata.json"

if not metadata_file.exists():
    sys.exit("kernel-metadata.json not found")

metadata = json.loads(
    metadata_file.read_text(
        encoding="utf-8"
    )
)

metadata["enable_gpu"] = True
metadata["machine_shape"] = "Gpu"


metadata_file.write_text(
    json.dumps(
        metadata,
        indent=2
    ),
    encoding="utf-8"
)


# Save modified notebook

notebook_file.write_text(
    json.dumps(
        notebook,
        indent=2
    ),
    encoding="utf-8"
)


print("\n[OK] GPU worker prepared", flush=True)


# ==================================================
# 10. Push to Kaggle
# ==================================================

print("\n[2] Pushing GPU Worker to Kaggle...", flush=True)

push_result = run(
    f"python -m kaggle kernels push "
    f"-p {WORK_DIR}",
    check=False
)

if push_result != 0:
    sys.exit(push_result)


print("\n[OK] Kaggle job submitted", flush=True)


# ==================================================
# 11. Wait for Kaggle
# ==================================================

print("\n[3] Waiting for Kaggle...", flush=True)

final_status = None

for i in range(120):

    print(
        f"\n========== STATUS {i + 1}/120 ==========",
        flush=True
    )

    result = subprocess.run(
        f"python -m kaggle kernels status {KERNEL}",
        shell=True,
        capture_output=True,
        text=True
    )

    status = (
        result.stdout +
        result.stderr
    )

    print(status, flush=True)

    upper = status.upper()

    if "COMPLETE" in upper:
        final_status = "COMPLETE"
        break

    if "ERROR" in upper:
        final_status = "ERROR"
        break

    if "FAILED" in upper:
        final_status = "FAILED"
        break

    if "CANCELLED" in upper:
        final_status = "CANCELLED"
        break

    time.sleep(10)


# ==================================================
# 12. Download output regardless of status
# ==================================================

print(
    "\n[4] Downloading Kaggle output...",
    flush=True
)

output_result = subprocess.run(
    f"python -m kaggle kernels output "
    f"{KERNEL} "
    f"-p {OUTPUT_DIR}",
    shell=True,
    text=True
)

print(
    f"Output command exit code: "
    f"{output_result.returncode}",
    flush=True
)


# ==================================================
# 13. Show files
# ==================================================

print("\n======================================")
print("           KAGGLE OUTPUT")
print("======================================")

files_found = False

for file in OUTPUT_DIR.rglob("*"):

    if file.is_file():

        files_found = True

        print(
            "OUTPUT:",
            file,
            flush=True
        )


if not files_found:
    print(
        "No output files were returned.",
        flush=True
    )


# ==================================================
# 14. Final result
# ==================================================

print("\n======================================")
print("           FINAL STATUS")
print("======================================")

print(
    "Kaggle status:",
    final_status,
    flush=True
)


if final_status != "COMPLETE":

    print(
        "\nKaggle Worker failed.",
        flush=True
    )

    print(
        "The output above should contain "
        "the useful error information.",
        flush=True
    )

    sys.exit(1)


print(
    "\nKaggle GPU Worker completed successfully.",
    flush=True
)
