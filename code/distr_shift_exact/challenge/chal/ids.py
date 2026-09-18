"""Random public identifiers for the released images.

The competition ships **no image files** (C9.5): a competitor gets the base
predictor's calibrated posterior per image and nothing else. Each image still
needs a key so that ``test_batches.csv`` can point at a row of
``predictions.csv``, and that key is a random 32-hex string.

The name hides which MedMNIST index an image came from. It hides nothing else,
and is not what protects the labels -- withholding the pixels is (C9.5). Two
batches drawing the same image still share an ``img_id``, which is visible and
is meant to be.
"""

from __future__ import annotations

import numpy as np


def random_ids(n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` distinct random 32-hex-character identifiers."""
    # 128 bits each: a collision over the ~66 000 ids used here has probability
    # around 1e-29, but it is checked rather than argued.
    raw = rng.integers(0, 1 << 32, size=(n, 4), dtype=np.uint64)
    ids = np.array(["".join(f"{v:08x}" for v in row) for row in raw])
    assert len(np.unique(ids)) == n, "random image ids collided"
    return ids
