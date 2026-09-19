"""The image transformation of the hard variant (tasks/hard_variant.md).

The hard variant publishes pixels, and TissueMNIST -- whose labels are public --
must be credited under its licence. Every released image is therefore
transformed before it leaves the organisers:

* rotated by a random multiple of 90 degrees -- 90, 180 or 270, never 0;
* given additive Gaussian noise of standard deviation ``noise_std`` grey
  levels, then rounded and clipped back to ``uint8``.

This defeats byte-exact matching (hashing) against the public archive, which in
the easy variant recovered every test label in seconds. It does **not** defeat
nearest-neighbour matching: noise small enough not to hurt a classifier is also
small enough for a nearest-neighbour search, so the rules are what forbid label
recovery. ``hard_audit_matching.py`` measures how much the transformation
actually buys.

The transformation is applied **per released row**, not per source image. A
test image drawn into two batches is released as two different arrays, so
identical pixels cannot link rows across batches.
"""

from __future__ import annotations

import numpy as np

QUARTER_TURNS = (1, 2, 3)          # 90, 180, 270 degrees
NOISE_STD_DEFAULT = 2.0            # grey levels, on the 0..255 scale
_CHUNK = 16384


def transform(X: np.ndarray, rng: np.random.Generator,
              noise_std: float = NOISE_STD_DEFAULT
              ) -> tuple[np.ndarray, np.ndarray]:
    """``(X_out uint8 (n, H, W), quarter_turns (n,))`` for ``X`` uint8 (n, H, W)."""
    X = np.asarray(X)
    assert X.dtype == np.uint8 and X.ndim == 3 and X.shape[1] == X.shape[2], \
        f"expected square uint8 images, got {X.dtype} {X.shape}"
    assert noise_std >= 0.0
    k = rng.choice(np.array(QUARTER_TURNS), size=len(X))
    out = np.empty_like(X)
    for q in QUARTER_TURNS:
        sel = k == q
        out[sel] = np.rot90(X[sel], q, axes=(1, 2))
    if noise_std > 0.0:
        # Chunked, in float32: a full float64 noise array for 126 000 rows
        # would be ~0.8 GB for no benefit.
        for a in range(0, len(out), _CHUNK):
            b = a + _CHUNK
            noisy = out[a:b].astype(np.float32) + noise_std * rng.standard_normal(
                out[a:b].shape, dtype=np.float32)
            out[a:b] = np.clip(np.rint(noisy), 0, 255).astype(np.uint8)
    return out, k
