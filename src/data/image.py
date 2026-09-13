from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".npy", ".png", ".tif", ".tiff"}


def read_array(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"image does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".npy":
        array = np.load(path, allow_pickle=False)
    elif suffix in {".tif", ".tiff"}:
        array = tifffile.imread(path)
    elif suffix in {".jpg", ".jpeg", ".png"}:
        with Image.open(path) as image:
            array = np.asarray(image)
    else:
        raise ValueError(f"unsupported image extension: {path.suffix}")
    if not isinstance(array, np.ndarray) or not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"image must contain a numeric array: {path}")
    return array


def write_mask(path: str | Path, masks: np.ndarray) -> None:
    path = Path(path)
    if masks.ndim != 2 or not np.issubdtype(masks.dtype, np.integer):
        raise ValueError("masks must be a two-dimensional integer array.")
    if np.any(masks < 0):
        raise ValueError("mask labels must be non-negative.")
    if path.suffix.lower() == ".png" and masks.max(initial=0) > 65535:
        raise ValueError("PNG supports labels up to 65535; use .npy or .tiff.")
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        np.save(path, masks, allow_pickle=False)
    elif suffix in {".tif", ".tiff"}:
        tifffile.imwrite(path, masks)
    elif suffix == ".png":
        Image.fromarray(masks.astype(np.uint16, copy=False)).save(path)
    else:
        raise ValueError("mask output must use .npy, .png, .tif, or .tiff.")


def find_pairs(
    image_dir: str | Path,
    mask_dir: str | Path,
) -> tuple[tuple[Path, Path], ...]:
    images = find_arrays(image_dir)
    masks = find_arrays(mask_dir)
    image_by_stem = index_by_stem(images, "image")
    mask_by_stem = index_by_stem(masks, "mask")
    missing_masks = sorted(set(image_by_stem) - set(mask_by_stem))
    missing_images = sorted(set(mask_by_stem) - set(image_by_stem))
    if missing_masks or missing_images:
        details = []
        if missing_masks:
            details.append(f"missing masks for {missing_masks}")
        if missing_images:
            details.append(f"missing images for {missing_images}")
        raise ValueError("image and mask stems do not match: " + "; ".join(details))
    return tuple(
        (image_by_stem[stem], mask_by_stem[stem]) for stem in sorted(image_by_stem)
    )


def find_arrays(folder: str | Path) -> tuple[Path, ...]:
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"data directory does not exist: {folder}")
    paths = tuple(
        sorted(
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    )
    if not paths:
        raise ValueError(f"data directory contains no supported arrays: {folder}")
    return paths


def index_by_stem(paths: tuple[Path, ...], kind: str) -> dict[str, Path]:
    indexed = {}
    for path in paths:
        if path.stem in indexed:
            raise ValueError(f"duplicate {kind} stem {path.stem!r}.")
        indexed[path.stem] = path
    return indexed
