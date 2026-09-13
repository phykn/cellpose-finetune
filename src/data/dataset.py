from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .. import MODEL_STRIDE
from ..prepare.convert import normalize_image
from ..prepare.flow import flow_targets
from .augment import crop_pair
from .image import find_pairs, read_array
from .label import class_target, read_labels


class CellposeDataset(Dataset):
    def __init__(
        self,
        image_dir: str | Path,
        mask_dir: str | Path,
        crop_size: int = 384,
        channel_axis: int | None = None,
        augment: bool = True,
        label_dir: str | Path | None = None,
        classes: tuple[str, ...] = (),
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
        self.classes = tuple(classes)
        self.label_dir = Path(label_dir) if label_dir is not None else None
        if bool(self.classes) != (self.label_dir is not None):
            raise ValueError(
                "classes and label_dir must be provided together for training."
            )
        if self.label_dir is not None:
            for image_path, _ in self.pairs:
                path = self.label_dir / f"{image_path.stem}.json"
                if not path.is_file():
                    raise FileNotFoundError(f"instance labels do not exist: {path}")

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

        labels = None
        if self.label_dir is not None:
            labels = read_labels(
                self.label_dir / f"{image_path.stem}.json", masks, self.classes
            )
        image, masks = crop_pair(
            image,
            masks,
            self.crop_size,
            augment=self.augment,
        )
        target = flow_targets(masks)
        if labels is not None:
            target = np.concatenate((target, class_target(masks, labels)[None]), axis=0)
        return (
            torch.from_numpy(image.copy()),
            torch.from_numpy(target.copy()),
        )
