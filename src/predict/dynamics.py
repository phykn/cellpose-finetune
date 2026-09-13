# Cellpose dynamics adapted from MouseLand/cellpose at a54cb488 (BSD-3-Clause).
# Copyright © 2025 Howard Hughes Medical Institute.

import numpy as np
import torch
from scipy.ndimage import binary_fill_holes, maximum_filter

from .. import FLOW_SCALE
from ..prepare.flow import _check_niter, _renumber, masks_to_flows


def follow_flows(
    flows: np.ndarray,
    cell_mask: np.ndarray,
    niter: int = 200,
    device: str | torch.device | None = None,
) -> np.ndarray:
    """Follow 2D flows from every foreground pixel with bilinear Euler steps."""
    flows = _check_flows(flows)
    cell_mask = _check_cell_mask(cell_mask, flows.shape[1:])
    niter = _check_niter(niter)
    inds = np.nonzero(cell_mask)
    if len(inds[0]) == 0:
        return np.zeros((2, 0), dtype=np.float32)

    torch_device = _as_device(device)
    field = torch.from_numpy(
        np.ascontiguousarray(flows * cell_mask[None], dtype=np.float32)
    ).to(torch_device)
    positions = torch.from_numpy(np.stack(inds).astype(np.float32, copy=False)).to(
        torch_device
    )
    height, width = cell_mask.shape

    for _ in range(niter):
        y = positions[0]
        x = positions[1]
        y0 = y.floor().to(torch.long)
        x0 = x.floor().to(torch.long)
        y1 = (y0 + 1).clamp(max=height - 1)
        x1 = (x0 + 1).clamp(max=width - 1)
        wy = y - y0
        wx = x - x0
        top = field[:, y0, x0] * (1.0 - wx) + field[:, y0, x1] * wx
        bottom = field[:, y1, x0] * (1.0 - wx) + field[:, y1, x1] * wx
        positions += top * (1.0 - wy) + bottom * wy
        positions[0].clamp_(0.0, float(height - 1))
        positions[1].clamp_(0.0, float(width - 1))

    return positions.cpu().numpy()


def compute_masks(
    flow_logits: np.ndarray,
    cellprob_logits: np.ndarray,
    niter: int = 200,
    cellprob_threshold: float = 0.0,
    flow_threshold: float | None = 0.4,
    min_size: int = 15,
    max_size_fraction: float = 0.4,
    device: str | torch.device | None = None,
) -> np.ndarray:
    """Reconstruct labelled masks from raw Cellpose flow and cell logits."""
    flow_logits = _check_flows(flow_logits, name="flow_logits")
    cellprob_logits = _check_logits(cellprob_logits, flow_logits.shape[1:])
    niter = _check_niter(niter)
    cellprob_threshold = _check_number(
        cellprob_threshold,
        "cellprob_threshold",
    )
    if flow_threshold is not None:
        flow_threshold = _check_number(flow_threshold, "flow_threshold")
        if flow_threshold < 0:
            raise ValueError("flow_threshold must be non-negative or None.")
    min_size = _check_min_size(min_size)
    max_size_fraction = _check_number(
        max_size_fraction,
        "max_size_fraction",
    )
    if not 0.0 < max_size_fraction <= 1.0:
        raise ValueError("max_size_fraction must be in (0, 1].")

    cell_mask = cellprob_logits > cellprob_threshold
    if not np.any(cell_mask):
        return np.zeros(cellprob_logits.shape, dtype=np.uint16)

    flows = flow_logits.astype(np.float32, copy=False) / FLOW_SCALE
    positions = follow_flows(
        flows,
        cell_mask,
        niter=niter,
        device=device,
    )
    masks = _masks_from_endpoints(
        positions,
        cell_mask,
        max_size_fraction,
    )
    if masks.max() > 0 and flow_threshold is not None and flow_threshold > 0:
        masks = _remove_bad_flow_masks(
            masks,
            flow_logits,
            flow_threshold,
        )
    masks = _fill_and_filter_masks(masks, min_size)
    dtype = np.uint16 if masks.max() < 2**16 else np.uint32
    return masks.astype(dtype, copy=False)


def _masks_from_endpoints(
    positions: np.ndarray,
    cell_mask: np.ndarray,
    max_size_fraction: float,
) -> np.ndarray:
    height, width = cell_mask.shape
    endpoints = positions.astype(np.int64)
    endpoints[0].clip(0, height - 1, out=endpoints[0])
    endpoints[1].clip(0, width - 1, out=endpoints[1])

    pad = 20
    y = endpoints[0] + pad
    x = endpoints[1] + pad
    hist = np.zeros((height + 2 * pad, width + 2 * pad), dtype=np.int32)
    np.add.at(hist, (y, x), 1)
    local_max = maximum_filter(hist, size=5, mode="constant")
    seeds = np.argwhere((hist == local_max) & (hist > 10))
    masks = np.zeros(cell_mask.shape, dtype=np.int32)
    if len(seeds) == 0:
        return masks

    counts = hist[seeds[:, 0], seeds[:, 1]]
    seeds = seeds[np.argsort(counts, kind="stable")]
    basins = np.zeros(hist.shape, dtype=np.int32)
    support = hist > 2
    for label, (seed_y, seed_x) in enumerate(seeds, start=1):
        region = np.s_[seed_y - 5 : seed_y + 6, seed_x - 5 : seed_x + 6]
        grown = np.zeros((11, 11), dtype=bool)
        grown[5, 5] = True
        for _ in range(5):
            grown = maximum_filter(grown, size=3, mode="constant")
            grown &= support[region]
        basins[region][grown] = label

    endpoint_labels = basins[y, x]
    masks[np.nonzero(cell_mask)] = endpoint_labels
    counts = np.bincount(masks.ravel())
    large = np.flatnonzero(counts > masks.size * max_size_fraction)
    large = large[large != 0]
    if len(large):
        masks[np.isin(masks, large)] = 0
    return _renumber(masks)


def _remove_bad_flow_masks(
    masks: np.ndarray,
    flow_logits: np.ndarray,
    threshold: float,
) -> np.ndarray:
    expected = masks_to_flows(masks)
    predicted = flow_logits.astype(np.float32, copy=False) / FLOW_SCALE
    error = np.square(expected - predicted).sum(axis=0)
    bad = []
    for label in range(1, int(masks.max()) + 1):
        value = error[masks == label].mean()
        if value > threshold:
            bad.append(label)
    if bad:
        masks = masks.copy()
        masks[np.isin(masks, bad)] = 0
    return _renumber(masks)


def _fill_and_filter_masks(masks: np.ndarray, min_size: int) -> np.ndarray:
    output = np.zeros(masks.shape, dtype=np.int32)
    next_label = 1
    for label in np.unique(masks):
        if label == 0:
            continue
        y, x = np.nonzero(masks == label)
        if min_size > 0 and len(y) < min_size:
            continue
        region = np.s_[y.min() : y.max() + 1, x.min() : x.max() + 1]
        filled = binary_fill_holes(masks[region] == label)
        if min_size > 0 and int(filled.sum()) < min_size:
            continue
        output[region][filled] = next_label
        next_label += 1
    return output


def _check_flows(flows: np.ndarray, name: str = "flows") -> np.ndarray:
    if not isinstance(flows, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array.")
    if flows.ndim != 3 or flows.shape[0] != 2:
        raise ValueError(f"{name} must have shape [2, H, W].")
    if 0 in flows.shape[1:]:
        raise ValueError(f"{name} must have non-empty spatial dimensions.")
    if not np.issubdtype(flows.dtype, np.number):
        raise TypeError(f"{name} must use a numeric dtype.")
    if not np.isfinite(flows).all():
        raise ValueError(f"{name} must contain only finite values.")
    return flows


def _check_cell_mask(cell_mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if not isinstance(cell_mask, np.ndarray):
        raise TypeError("cell_mask must be a NumPy array.")
    if cell_mask.shape != shape:
        raise ValueError("cell_mask shape must match flows spatial shape.")
    if not (
        np.issubdtype(cell_mask.dtype, np.bool_)
        or np.issubdtype(cell_mask.dtype, np.number)
    ):
        raise TypeError("cell_mask must use a boolean or numeric dtype.")
    if np.issubdtype(cell_mask.dtype, np.number) and not np.isfinite(cell_mask).all():
        raise ValueError("cell_mask must contain only finite values.")
    return cell_mask.astype(bool, copy=False)


def _check_logits(logits: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if not isinstance(logits, np.ndarray):
        raise TypeError("cellprob_logits must be a NumPy array.")
    if logits.shape != shape:
        raise ValueError("cellprob_logits shape must match flow_logits spatial shape.")
    if not np.issubdtype(logits.dtype, np.number):
        raise TypeError("cellprob_logits must use a numeric dtype.")
    if not np.isfinite(logits).all():
        raise ValueError("cellprob_logits must contain only finite values.")
    return logits


def _check_min_size(min_size: int) -> int:
    if not isinstance(min_size, (int, np.integer)) or isinstance(min_size, bool):
        raise TypeError("min_size must be an integer.")
    if min_size < -1:
        raise ValueError("min_size must be at least -1.")
    return int(min_size)


def _check_number(value: float, name: str) -> float:
    if not isinstance(value, (int, float, np.integer, np.floating)) or isinstance(
        value,
        bool,
    ):
        raise TypeError(f"{name} must be a finite number.")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number.")
    return value


def _as_device(device: str | torch.device | None) -> torch.device:
    try:
        return torch.device("cpu" if device is None else device)
    except (RuntimeError, TypeError) as exc:
        raise ValueError(f"invalid torch device: {device!r}") from exc
