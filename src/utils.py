from pathlib import Path

import yaml


def load_yaml(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    if not isinstance(cfg, dict):
        raise TypeError(f"YAML root must be a mapping: {path}")
    return cfg


def save_yaml(path: str | Path, cfg: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(cfg, file, sort_keys=False)
