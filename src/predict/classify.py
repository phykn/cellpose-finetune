import numpy as np
from scipy.special import softmax


def classify_instances(
    masks: np.ndarray, logits: np.ndarray, classes: tuple[str, ...]
) -> tuple[dict, ...]:
    if logits.shape != (len(classes), *masks.shape) or not classes:
        raise ValueError(
            "class logits must have shape [classes, mask height, mask width]."
        )
    if not np.isfinite(logits).all():
        raise ValueError("class logits must contain only finite values.")
    probs = softmax(logits, axis=0)
    instances = []
    for mask_id in np.unique(masks):
        if mask_id == 0:
            continue
        scores = probs[:, masks == mask_id].mean(axis=1)
        class_id = int(scores.argmax())
        instances.append(
            {
                "mask_id": int(mask_id),
                "class_id": class_id,
                "class_name": classes[class_id],
                "score": float(scores[class_id]),
            }
        )
    return tuple(instances)
