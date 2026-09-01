"""Verify the training environment. Run after `pip install -r requirements.txt`.

    python scripts/check_env.py
"""
from __future__ import annotations

import os
import sys


def main() -> int:
    ok = True

    print(f"python      : {sys.version.split()[0]}  ({sys.executable})")

    try:
        import torch

        print(f"torch       : {torch.__version__}")
        cuda = torch.cuda.is_available()
        print(f"cuda avail  : {cuda}")
        if cuda:
            print(f"device      : {torch.cuda.get_device_name(0)}")
            print(f"capability  : {torch.cuda.get_device_capability(0)}")
            bf16 = torch.cuda.is_bf16_supported()
            print(f"bf16        : {bf16}")
            ok &= bf16
            x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
            _ = (x @ x).float().sum().item()
            print("gpu matmul  : OK")
        else:
            ok = False
    except Exception as e:  # noqa: BLE001
        print(f"torch       : FAIL - {e!r}")
        ok = False

    for mod in ["tokenizers", "transformers", "datasets", "accelerate",
                "numpy", "pandas", "pydantic", "yaml", "wandb", "datasketch", "ftfy"]:
        try:
            m = __import__(mod)
            print(f"{mod:<12}: {getattr(m, '__version__', 'ok')}")
        except Exception as e:  # noqa: BLE001
            print(f"{mod:<12}: MISSING - {e!r}")
            ok = False

    for var in ["PIP_CACHE_DIR", "HF_HOME", "HF_DATASETS_CACHE", "TORCH_HOME", "WANDB_DIR"]:
        val = os.environ.get(var, "<unset>")
        on_d = val.upper().startswith("D:")
        print(f"env {var:<18}: {val}  {'OK' if on_d else '!! not on D:'}")
        ok &= on_d

    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
