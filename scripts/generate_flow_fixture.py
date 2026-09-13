"""Generate offline flow references from an inspected upstream dynamics.py."""

import argparse
import ast
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import find_objects


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    names = {
        "_extend_centers_gpu",
        "center_of_mass",
        "get_centers",
        "masks_to_flows_gpu",
    }
    tree = ast.parse(args.source.read_text(encoding="utf-8"))
    selected = ast.Module(
        body=[
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in names
        ],
        type_ignores=[],
    )
    scope = {"np": np, "torch": torch, "F": F, "find_objects": find_objects}
    exec(compile(selected, str(args.source), "exec"), scope)
    masks = np.zeros((4, 32, 40), dtype=np.int32)
    masks[0, 1:7, 3:9] = 1
    masks[0, 18:27, 24:35] = 2
    masks[1, 1:12, 1:5] = 1
    masks[1, 8:12, 1:16] = 1
    masks[2, 0:6, 0:4] = 1
    masks[2, 26:32, 34:40] = 2
    flows = [
        scope["masks_to_flows_gpu"](mask, device=torch.device("cpu"))[0]
        for mask in masks
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, masks=masks, flows=np.asarray(flows, dtype=np.float32)
    )


if __name__ == "__main__":
    main()
