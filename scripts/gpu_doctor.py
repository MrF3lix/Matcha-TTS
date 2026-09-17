"""Report why PyTorch can or cannot see a GPU on this node.

Run it standalone on a compute node to debug an allocation:

    srun --partition=gpu --gres=gpu:2 --account=cai_nlp uv run scripts/gpu_doctor.py

train.submit also runs it before training, so a bad node fails in the first seconds
with an explanation instead of silently falling back to CPU for the whole time limit.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

# SLURM exports a different one of these depending on the vendor and the site config;
# an empty value (as opposed to unset) is what silently hides every GPU.
GPU_ENV_VARS = [
    "CUDA_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
    "SLURM_JOB_GPUS",
    "SLURM_STEP_GPUS",
    "SLURM_GPUS_ON_NODE",
]


def section(title):
    print(f"\n--- {title} " + "-" * max(0, 60 - len(title)))


def run(cmd):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return (out.stdout + out.stderr).strip()
    except Exception as exc:  # noqa: BLE001 - diagnostics must never crash
        return f"<failed: {exc}>"


def main():
    section("node")
    print("host:", os.uname().nodename)
    print("python:", sys.executable)

    section("gpu environment")
    for var in GPU_ENV_VARS:
        val = os.environ.get(var)
        # Distinguish unset from set-but-empty: the latter hides all devices.
        print(f"{var} = {'<unset>' if val is None else repr(val)}")

    section("device nodes")
    # Under a SLURM cgroup these only appear if the step was actually granted the GPUs.
    print("/dev/kfd exists:", Path("/dev/kfd").exists(), "(AMD compute device)")
    dri = sorted(str(p) for p in Path("/dev/dri").glob("renderD*")) if Path("/dev/dri").is_dir() else []
    print("/dev/dri render nodes:", dri or "<none>")
    nvidia = sorted(str(p) for p in Path("/dev").glob("nvidia[0-9]*"))
    print("/dev/nvidia* :", nvidia or "<none>")

    section("vendor tools")
    if shutil.which("nvidia-smi"):
        print(run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv"]))
    else:
        print("nvidia-smi: not on PATH")
    rocm_version = Path("/opt/rocm/.info/version")
    print("ROCm version file:", rocm_version.read_text().strip() if rocm_version.exists() else "<absent>")
    if shutil.which("rocm-smi"):
        print(run(["rocm-smi", "--showproductname"]))
    else:
        print("rocm-smi: not on PATH")

    section("torch")
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        print("FAILED to import torch:", exc)
        return 1

    print("torch version :", torch.__version__)
    print("built for cuda:", torch.version.cuda)
    print("built for hip :", torch.version.hip)

    wheel = "ROCm" if torch.version.hip else ("CUDA" if torch.version.cuda else "CPU-only")
    print("wheel variant :", wheel)

    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        print(f"\nOK: {count} GPU(s) visible to torch")
        for i in range(count):
            print(f"  [{i}] {torch.cuda.get_device_name(i)}")
        return 0

    # is_available() swallows the underlying error; init() raises it.
    print("\nFAIL: torch.cuda.is_available() is False")
    try:
        torch.cuda.init()
    except Exception as exc:  # noqa: BLE001
        print("underlying error:", type(exc).__name__, exc)

    print("\nLikely causes, in the order worth checking:")
    if wheel == "CPU-only":
        print("  * A CPU-only torch wheel got installed -- the accelerator extra did not apply.")
    elif wheel == "ROCm" and not Path("/dev/kfd").exists():
        print("  * /dev/kfd is missing: this node has no AMD GPU, or the SLURM cgroup did")
        print("    not grant the step access to it.")
    elif wheel == "ROCm":
        print("  * ROCm version skew: the wheels are built for the rocm index pinned in")
        print("    pyproject.toml; compare against the ROCm version printed above.")
        print("  * Unsupported GPU architecture -- try HSA_OVERRIDE_GFX_VERSION.")
    elif wheel == "CUDA" and not nvidia:
        print("  * A CUDA wheel is installed but this node exposes no /dev/nvidia*.")
        print("    If the node is AMD, vendor detection in train.submit picked wrong.")
    else:
        print("  * Driver too old for this CUDA build, or the devices were not granted")
        print("    to this step (check the visible-device variables above).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
