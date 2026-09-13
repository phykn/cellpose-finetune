import argparse
import json
from pathlib import Path

import torch

from src.build import build_model
from src.config import find_prediction_config, load_config
from src.data.image import read_array, write_mask
from src.predict.inference import predict


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default=None,
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    config = (
        args.config if args.config is not None else find_prediction_config(args.weights)
    )
    cfg = load_config(config, device=args.device)
    device = torch.device(cfg["device"])
    model = build_model(cfg, weights=args.weights, load_backbone=False).to(device)
    result = predict(
        model,
        read_array(args.image),
        device=device,
        channel_axis=cfg["data"].get("channel_axis"),
        classes=tuple(cfg["model"]["classes"]),
        **cfg["predict"],
    )
    write_mask(args.output, result.masks)
    if cfg["model"]["classes"]:
        label_output = args.output.with_suffix(".labels.json")
        with label_output.open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "format_version": 1,
                    "classes": cfg["model"]["classes"],
                    "instances": result.instances,
                },
                file,
                ensure_ascii=False,
                indent=2,
            )


if __name__ == "__main__":
    main()
