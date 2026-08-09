import argparse
from pathlib import Path

import torch

from src.build import build_model
from src.dataset import read_array, write_mask
from src.inference import predict
from src.utils import load_yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config" / "train.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    cfg = load_yaml(args.config)
    model = build_model(cfg, weights=args.weights, load_backbone=False).to(device)
    result = predict(
        model,
        read_array(args.image),
        device=device,
        channel_axis=cfg["data"].get("channel_axis"),
    )
    write_mask(args.output, result.masks)


if __name__ == "__main__":
    main()
