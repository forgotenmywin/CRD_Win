import os
import sys
import time
import shutil
import subprocess
from pathlib import Path


# =========================================================
# CONFIG
# =========================================================

KAGGLE_USERNAME = os.environ.get("KAGGLE_USERNAME")
KAGGLE_KERNEL = f"{KAGGLE_USERNAME}/notebookd9d7092a0a"

KERNEL_DIR = Path("kaggle_kernel")
OUTPUT_DIR = Path("kaggle_output")

KERNEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# =========================================================
# HELPERS
# =========================================================

def run(command, check=True):

    print()
    print(">>>", " ".join(command))

    result = subprocess.run(
        command,
        text=True
    )

    if check and result.returncode != 0:
        print("Command failed.")
        sys.exit(result.returncode)

    return result.returncode


def kaggle(*args, check=True):

    return run(
        ["python", "-m", "kaggle"] + list(args),
        check=check
    )


# =========================================================
# START
# =========================================================

print("======================================")
print("      KAGGLE GPU WORKER")
print("======================================")


if not KAGGLE_USERNAME:
    print("ERROR: KAGGLE_USERNAME missing")
    sys.exit(1)


# =========================================================
# 1. CLEAN
# =========================================================

if KERNEL_DIR.exists():
    shutil.rmtree(KERNEL_DIR)

KERNEL_DIR.mkdir()

print()
print("[1] Downloading Kaggle notebook...")


# =========================================================
# 2. PULL NOTEBOOK
# =========================================================

kaggle(
    "kernels",
    "pull",
    KAGGLE_KERNEL,
    "-p",
    str(KERNEL_DIR),
    "-m"
)


# =========================================================
# 3. CHECK METADATA
# =========================================================

metadata_file = (
    KERNEL_DIR /
    "kernel-metadata.json"
)

if not metadata_file.exists():

    print("ERROR: kernel-metadata.json not found")
    sys.exit(1)


print()
print("[2] Checking Kaggle metadata...")

metadata = metadata_file.read_text(
    encoding="utf-8"
)

print(metadata)


# =========================================================
# 4. FIND NOTEBOOK
# =========================================================

notebooks = list(
    KERNEL_DIR.glob("*.ipynb")
)

if not notebooks:

    print("ERROR: notebook not found")
    sys.exit(1)

notebook = notebooks[0]

print()
print("[3] Notebook:")
print(notebook)


# =========================================================
# 5. INJECT JOB
# =========================================================

print()
print("[4] Injecting dynamic GPU job...")


import json

data = json.loads(
    notebook.read_text(
        encoding="utf-8"
    )
)


job_file = Path("job.py")

if not job_file.exists():

    print("ERROR: job.py not found")
    sys.exit(1)


job_code = job_file.read_text(
    encoding="utf-8"
)


cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": job_code.splitlines(
        keepends=True
    )
}


data["cells"].append(cell)


notebook.write_text(
    json.dumps(
        data,
        ensure_ascii=False,
        indent=1
    ),
    encoding="utf-8"
)


print("Dynamic job injected.")


# =========================================================
# 6. PUSH TO KAGGLE
# =========================================================

print()
print("[5] Pushing notebook to Kaggle...")


kaggle(
    "kernels",
    "push",
    "-p",
    str(KERNEL_DIR)
)


# =========================================================
# 7. WAIT FOR GPU
# =========================================================

print()
print("======================================")
print("       WAITING FOR KAGGLE GPU")
print("======================================")


MAX_CHECKS = 120

final_status = None


for i in range(1, MAX_CHECKS + 1):

    print()
    print(
        f"========== STATUS {i}/{MAX_CHECKS} =========="
    )

    result = subprocess.run(
        [
            "python",
            "-m",
            "kaggle",
            "kernels",
            "status",
            KAGGLE_KERNEL
        ],
        text=True,
        capture_output=True
    )

    output = (
        result.stdout +
        result.stderr
    )

    print(output.strip())

    if "COMPLETE" in output:

        final_status = "COMPLETE"
        break

    if "ERROR" in output:

        final_status = "ERROR"
        break

    if "CANCEL" in output:

        final_status = "CANCELLED"
        break

    time.sleep(5)


# =========================================================
# 8. DOWNLOAD OUTPUT
# =========================================================

print()
print("======================================")
print("       DOWNLOADING KAGGLE OUTPUT")
print("======================================")


kaggle(
    "kernels",
    "output",
    KAGGLE_KERNEL,
    "-p",
    str(OUTPUT_DIR),
    check=False
)


# =========================================================
# 9. FINAL
# =========================================================

print()
print("======================================")
print("           FINAL STATUS")
print("======================================")

print(
    "Kaggle status:",
    final_status
)


if final_status != "COMPLETE":

    print()
    print("GPU Worker failed.")

    sys.exit(1)


print()
print("GPU Worker completed successfully.")
