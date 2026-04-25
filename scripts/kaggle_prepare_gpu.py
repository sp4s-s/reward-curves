#!/usr/bin/env python3
"""Prepare Kaggle GPU runtimes for dpocurv."""
from __future__ import annotations

import argparse
import subprocess
import sys

P100_TORCH = [
    "torch==2.4.1",
    "torchvision==0.19.1",
    "torchaudio==2.4.1",
    "--index-url",
    "https://download.pytorch.org/whl/cu121",
]


def torch_info():
    try:
        import torch
    except Exception as exc:
        return f"torch import failed: {exc}", None, []
    archs = getattr(torch.cuda, "get_arch_list", lambda: [])()
    if not torch.cuda.is_available():
        return f"torch={torch.__version__}; cuda unavailable; archs={archs}", None, archs
    cap = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    return f"torch={torch.__version__}; gpu={name}; capability=sm_{cap[0]}{cap[1]}; archs={archs}", cap, archs


def supports(cap, archs):
    if cap is None or not archs:
        return True
    return f"sm_{cap[0]}{cap[1]}" in archs or f"compute_{cap[0]}{cap[1]}" in archs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--install", action="store_true", help="Install a P100-compatible PyTorch wheel when needed.")
    args = parser.parse_args()

    info, cap, archs = torch_info()
    print(info)
    if supports(cap, archs):
        print("GPU/PyTorch compatibility looks OK.")
        return 0

    print("\nCurrent PyTorch does not support this GPU architecture.")
    if cap == (6, 0):
        print("Detected P100/sm_60. Recommended Kaggle fix:")
        print("  !python scripts/kaggle_prepare_gpu.py --install")
        print("  !python src/dpocurv/cli/main.py experiment=dpo_curv wandb.enabled=false")
        if args.install:
            cmd = [sys.executable, "-m", "pip", "install", "--force-reinstall", *P100_TORCH]
            print("\nRunning:", " ".join(cmd))
            subprocess.check_call(cmd)
            print("\nInstalled P100-compatible PyTorch. Run training in the next cell/process.")
            return 0
    else:
        print("Switch Kaggle accelerator to T4/V100/H100, or install a PyTorch wheel supporting this GPU.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
