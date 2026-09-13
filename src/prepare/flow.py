# Cellpose dynamics adapted from MouseLand/cellpose at a54cb488 (BSD-3-Clause).
# Copyright © 2025 Howard Hughes Medical Institute.

import numpy as np
from scipy.ndimage import find_objects

NEIGHBOR_OFFSETS = np.array(
    (
        (0, 0),
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ),
    dtype=np.intp,
)


def masks_to_flows(
    masks: np.ndarray,
    niter: int | None = None,
) -> np.ndarray:
    """Convert labelled 2D masks to normalized inward Y/X flows."""
    masks = _check_masks(masks)
    niter = _check_niter(niter, allow_none=True)
    labels = _renumber(masks)
    flows = np.zeros((2, *labels.shape), dtype=np.float32)
    if not np.any(labels):
        return flows

    padded = np.pad(labels, 1)
    y, x = np.nonzero(padded)
    neighbors_y = y[None] + NEIGHBOR_OFFSETS[:, 0, None]
    neighbors_x = x[None] + NEIGHBOR_OFFSETS[:, 1, None]
    values = padded[y, x]
    is_neighbor = padded[neighbors_y, neighbors_x] == values[None]

    centers, extents = _mask_centers(labels)
    centers += 1
    steps = 2 * int(extents.max()) if niter is None else niter
    heat = np.zeros(padded.shape, dtype=np.float64)
    for _ in range(steps):
        heat[centers[:, 0], centers[:, 1]] += 1.0
        neighbor_heat = heat[neighbors_y, neighbors_x]
        heat[y, x] = (neighbor_heat * is_neighbor).sum(axis=0) / len(NEIGHBOR_OFFSETS)

    dy = heat[neighbors_y[2], neighbors_x[2]] - heat[neighbors_y[1], neighbors_x[1]]
    dx = heat[neighbors_y[4], neighbors_x[4]] - heat[neighbors_y[3], neighbors_x[3]]
    vectors = np.stack((dy, dx))
    magnitude = np.linalg.norm(vectors, axis=0)
    nonzero = magnitude > 0
    vectors[:, nonzero] /= magnitude[nonzero]
    flows[:, y - 1, x - 1] = vectors.astype(np.float32, copy=False)
    return flows


def flow_targets(masks: np.ndarray) -> np.ndarray:
    """Return Cellpose targets ordered as cell probability, Y flow, X flow."""
    flows = masks_to_flows(masks)
    cellprob = (masks > 0).astype(np.float32, copy=False)
    return np.concatenate((cellprob[None], flows), axis=0)


def _mask_centers(masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centers = []
    extents = []
    for label, region in enumerate(find_objects(masks), start=1):
        y, x = np.nonzero(masks[region] == label)
        mean_y = int(np.rint(y.mean()))
        mean_x = int(np.rint(x.mean()))
        at_mean = np.flatnonzero((y == mean_y) & (x == mean_x))
        if len(at_mean):
            idx = int(at_mean[0])
        else:
            idx = int(np.argmin((y - mean_y) ** 2 + (x - mean_x) ** 2))
        # Upstream rounds within each bounding box; rounding global coordinates
        # changes half-integer ties for boxes starting on odd coordinates.
        centers.append((y[idx] + region[0].start, x[idx] + region[1].start))
        extents.append(int(y.max() - y.min() + x.max() - x.min() + 4))
    return np.asarray(centers, dtype=np.intp), np.asarray(extents, dtype=np.intp)


def _renumber(masks: np.ndarray) -> np.ndarray:
    values, inverse = np.unique(masks, return_inverse=True)
    labels = np.zeros(len(values), dtype=np.int32)
    labels[values > 0] = np.arange(1, np.count_nonzero(values > 0) + 1)
    return labels[inverse].reshape(masks.shape)


def _check_masks(masks: np.ndarray) -> np.ndarray:
    if not isinstance(masks, np.ndarray):
        raise TypeError("masks must be a NumPy array.")
    if masks.ndim != 2:
        raise ValueError("masks must have shape [H, W].")
    if 0 in masks.shape:
        raise ValueError("masks must have non-empty spatial dimensions.")
    if not np.issubdtype(masks.dtype, np.integer):
        raise TypeError("masks must use an integer dtype.")
    if np.any(masks < 0):
        raise ValueError("masks must contain non-negative labels.")
    return masks


def _check_niter(
    niter: int | None,
    allow_none: bool = False,
) -> int | None:
    if niter is None and allow_none:
        return None
    if not isinstance(niter, (int, np.integer)) or isinstance(niter, bool):
        raise TypeError("niter must be a non-negative integer.")
    if niter < 0:
        raise ValueError("niter must be a non-negative integer.")
    return int(niter)
