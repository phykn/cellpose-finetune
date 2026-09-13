import argparse
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from src.build import build_trainer
from src.config import DEFAULT_CONFIG, PROJECT_ROOT, load_config, save_yaml

RUN_ROOT = PROJECT_ROOT / "run"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default=None,
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--weights", type=Path, default=None)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    cfg = load_config(
        args.config, device=args.device, resume=args.resume, weights=args.weights
    )
    device = torch.device(cfg["device"])
    seed = cfg["train"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    trainer = build_trainer(cfg, device)
    run_dir = make_run_dir(RUN_ROOT) if args.run_dir is None else args.run_dir
    run_dir.mkdir(parents=True, exist_ok=False)
    save_yaml(run_dir / "train.yaml", cfg)
    trainer.fit(
        steps=cfg["train"]["total_steps"],
        save_every=cfg["train"]["save_every_steps"],
        run_dir=run_dir,
    )


def make_run_dir(root: Path) -> Path:
    name = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    for sequence in range(1, 100):
        suffix = "" if sequence == 1 else f"{sequence:02d}"
        path = root / f"{name}{suffix}"
        if not path.exists():
            return path
    raise FileExistsError(f"too many runs already exist for timestamp {name}.")


if __name__ == "__main__":
    main()
