from pathlib import Path

import numpy as np

ANGLE_DEGREES = np.arange(-45.0, 46.0, 5.0)


def save_model(path, model, target_name, min_k, max_k, **metadata):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        target_name=target_name,
        min_k=min_k,
        max_k=max_k,
        **metadata,
        w1=model["w1"],
        b1=model["b1"],
        w2=model["w2"],
        b2=model["b2"],
    )
