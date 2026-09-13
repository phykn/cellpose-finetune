import numpy as np
import torch


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
