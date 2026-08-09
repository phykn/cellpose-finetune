from pathlib import Path

import numpy as np
import tifffile
import torch
from PIL import Image
from torch.utils.data import Dataset

from . import MODEL_STRIDE
from .dynamics import flow_targets
from .transforms import normalize_image

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


class CellposeDataset(Dataset):
    def __init__(
        self,
        image_dir: str | Path,
        mask_dir: str | Path,
        crop_size: int = 384,
        channel_axis: int | None = None,
        augment: bool = True,
    ) -> None:
        if not isinstance(crop_size, int) or isinstance(crop_size, bool):
            raise TypeError("crop_size must be an integer.")
        if crop_size < 16 or crop_size % MODEL_STRIDE:
            raise ValueError(
                f"crop_size must be at least 16 and divisible by {MODEL_STRIDE}."
            )
        if not isinstance(augment, bool):
            raise TypeError("augment must be a boolean.")
        self.pairs = find_pairs(image_dir, mask_dir)
        self.crop_size = crop_size
        self.channel_axis = channel_axis
        self.augment = augment

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.pairs[idx]
        image = normalize_image(
            read_array(image_path),
            channel_axis=self.channel_axis,
        )
        masks = read_array(mask_path)
        if masks.ndim != 2:
            raise ValueError(f"instance mask must be two-dimensional: {mask_path}")
        integer_labels = (
            np.issubdtype(masks.dtype, np.integer)
            or np.equal(
                masks,
                np.floor(masks),
            ).all()
        )
        if not integer_labels:
            raise TypeError(f"instance mask must contain integer labels: {mask_path}")
        masks = masks.astype(np.int64, copy=False)
        if masks.min(initial=0) < 0:
            raise ValueError(f"instance mask labels must be non-negative: {mask_path}")
        if image.shape[-2:] != masks.shape:
            raise ValueError(
                f"image and mask shapes do not match: {image_path}, {mask_path}"
            )

        image, masks = crop_pair(
            image,
            masks,
            self.crop_size,
            augment=self.augment,
        )
        target = flow_targets(masks)
        return (
            torch.from_numpy(image.copy()),
            torch.from_numpy(target.copy()),
        )


def crop_pair(
    image: np.ndarray,
    masks: np.ndarray,
    size: int,
    augment: bool,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = masks.shape
    pad_height = max(0, size - height)
    pad_width = max(0, size - width)
    if pad_height or pad_width:
        image = np.pad(image, ((0, 0), (0, pad_height), (0, pad_width)))
        masks = np.pad(masks, ((0, pad_height), (0, pad_width)))
        height, width = masks.shape

    if augment:
        top = int(torch.randint(0, height - size + 1, ()).item())
        left = int(torch.randint(0, width - size + 1, ()).item())
    else:
        top = (height - size) // 2
        left = (width - size) // 2
    image = image[:, top : top + size, left : left + size]
    masks = masks[top : top + size, left : left + size]

    if augment:
        turns = int(torch.randint(0, 4, ()).item())
        image = np.rot90(image, turns, axes=(-2, -1))
        masks = np.rot90(masks, turns)
        if bool(torch.randint(0, 2, ()).item()):
            image = image[:, :, ::-1]
            masks = masks[:, ::-1]
        if bool(torch.randint(0, 2, ()).item()):
            image = image[:, ::-1, :]
            masks = masks[::-1, :]
    return image, masks
