"""Check every floating-point tensor in a checkpoint for NaN or infinity."""

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    invalid = [
        name
        for name, value in state_dict.items()
        if torch.is_tensor(value) and (value.is_floating_point() or value.is_complex())
        and not torch.isfinite(value).all().item()
    ]
    if invalid:
        for name in invalid:
            print(f"Non-finite values found in {name}")
        raise SystemExit(1)
    print(f"All floating-point tensors are finite: {args.checkpoint}")


if __name__ == "__main__":
    main()
