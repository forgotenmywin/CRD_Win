import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


# ==================================================
# CONFIG
# ==================================================

USERNAME = os.environ["KAGGLE_USERNAME"]
KERNEL = f"{USERNAME}/notebookd9d7092a0a"

WORK_DIR = Path("kaggle_kernel")
OUTPUT_DIR = Path("kaggle_output")


# ==================================================
# COMMAND RUNNER
# ==================================================

def run(command, check=True):

    print(f"\n>>> {command}", flush=True)

    result = subprocess.run(
        command,
        shell=True,
        text=True
    )

    if check and result.returncode != 0:
        print(
            f"Command failed: {result.returncode}",
            flush=True
        )

        sys.exit(result.returncode)

    return result.returncode


# ==================================================
# START
# ==================================================

print("======================================")
print("       KAGGLE GPU GENERIC WORKER")
print("======================================")

print(
    "Kaggle user:",
    USERNAME,
    flush=True
)


# ==================================================
# CHECK CREDENTIALS
# ==================================================

if not os.getenv("KAGGLE_API_TOKEN"):
    sys.exit("KAGGLE_API_TOKEN is missing")

print("[OK] Kaggle credentials found")


# ==================================================
# CLEAN
# ==================================================

if WORK_DIR.exists():
    shutil.rmtree(WORK_DIR)

if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)

WORK_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==================================================
# PULL NOTEBOOK
# ==================================================

print("\n[1] Pulling Kaggle Notebook...")

run(
    f"python -m kaggle kernels pull "
    f"{KERNEL} "
    f"-p {WORK_DIR} "
    f"-m"
)


# ==================================================
# FIND NOTEBOOK
# ==================================================

notebook_files = list(
    WORK_DIR.glob("*.ipynb")
)

if not notebook_files:
    sys.exit(
        "No .ipynb file found"
    )

notebook_file = notebook_files[0]

print(
    "[OK] Notebook:",
    notebook_file,
    flush=True
)


# ==================================================
# READ JOB
# ==================================================

job_file = Path("job.py")

if not job_file.exists():
    sys.exit(
        "job.py not found"
    )

job_code = job_file.read_text(
    encoding="utf-8"
)

print(
    "[OK] job.py loaded",
    flush=True
)


# ==================================================
# LOAD NOTEBOOK
# ==================================================

notebook = json.loads(
    notebook_file.read_text(
        encoding="utf-8"
    )
)


# ==================================================
# CREATE WORKER CELL
# ==================================================

worker_code = """
print("======================================")
print("       KAGGLE GPU WORKER START")
print("======================================")

import sys

print("Python:", sys.version)

try:

    import torch

    print(
        "PyTorch:",
        torch.__version__
    )

    print(
        "CUDA available:",
        torch.cuda.is_available()
    )

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA GPU is NOT available"
        )

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "CUDA version:",
        torch.version.cuda
    )

except Exception as e:

    print("GPU CHECK ERROR")
    print(
        type(e).__name__,
        str(e)
    )

    raise


print("")
print("======================================")
print("          RUNNING USER JOB")
print("======================================")

""" + job_code + """

print("")
print("======================================")
print("          USER JOB FINISHED")
print("======================================")
"""


# ==================================================
# REPLACE NOTEBOOK CELLS
# ==================================================

notebook["cells"] = [
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": worker_code.splitlines(
            keepends=True
        )
    }
]


# ==================================================
# UPDATE METADATA
# ==================================================

metadata_file = (
    WORK_DIR /
    "kernel-metadata.json"
)

if not metadata_file.exists():

    sys.exit(
        "kernel-metadata.json not found"
    )


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


# ==================================================
# SAVE NOTEBOOK
# ==================================================

notebook_file.write_text(
    json.dumps(
        notebook,
        indent=2
    ),
    encoding="utf-8"
)


print(
    "\n[OK] Notebook prepared",
    flush=True
)


# ==================================================
# PUSH
# ==================================================

print(
    "\n[2] Pushing to Kaggle...",
    flush=True
)

push_result = run(
    f"python -m kaggle kernels push "
    f"-p {WORK_DIR}",
    check=False
)

if push_result != 0:

    sys.exit(
        push_result
    )


print(
    "\n[OK] Kaggle job submitted",
    flush=True
)


# ==================================================
# WAIT FOR KAGGLE
# ==================================================

print(
    "\n[3] Waiting for Kaggle...",
    flush=True
)

final_status = None


for i in range(120):

    print("")
    print(
        "======================================"
    )
    print(
        f"STATUS {i + 1}/120"
    )
    print(
        "======================================"
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

    print(
        status,
        flush=True
    )

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
# DOWNLOAD OUTPUT / LOG
# ==================================================

print(
    "\n[4] Downloading Kaggle output...",
    flush=True
)

subprocess.run(
    f"python -m kaggle kernels output "
    f"{KERNEL} "
    f"-p {OUTPUT_DIR}",
    shell=True,
    text=True
)


# ==================================================
# LIST OUTPUT
# ==================================================

print("")
print(
    "======================================"
)

print(
    "           KAGGLE OUTPUT"
)

print(
    "======================================"
)


files = list(
    OUTPUT_DIR.rglob("*")
)


if not files:

    print(
        "No files returned."
    )

else:

    for file in files:

        if file.is_file():

            print(
                "OUTPUT:",
                file,
                flush=True
            )


# ==================================================
# PRINT LOG CONTENT
# ==================================================

print("")
print(
    "======================================"
)

print(
    "           KAGGLE LOG"
)

print(
    "======================================"
)


log_files = list(
    OUTPUT_DIR.rglob("*.log")
)


if not log_files:

    print(
        "No .log file found."
    )

else:

    for log_file in log_files:

        print("")
        print(
            f"========== {log_file} =========="
        )

        try:

            content = log_file.read_text(
                encoding="utf-8",
                errors="replace"
            )

            print(
                content,
                flush=True
            )

        except Exception as e:

            print(
                "Could not read log:",
                e,
                flush=True
            )


# ==================================================
# FINAL STATUS
# ==================================================

print("")
print(
    "======================================"
)

print(
    "           FINAL STATUS"
)

print(
    "======================================"
)

print(
    "Kaggle:",
    final_status,
    flush=True
)


if final_status != "COMPLETE":

    print(
        "\nGPU Worker failed.",
        flush=True
    )

    sys.exit(1)


print(
    "\nGPU Worker completed successfully.",
    flush=True
            )
