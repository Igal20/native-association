"""Coordinate quantization helpers (public re-implementation).

Internally these live in a shared production utility module; the public
version reproduces the exact behaviour used by the paper: normalized
[0, 1] coordinates are quantized to ``loc_bins`` bins and rendered as
``<loc_i>`` tokens (paper Sec. 3.1).
"""

from __future__ import annotations

from typing import Sequence


def quantize_coord(value: float, loc_bins: int) -> int:
    """Quantize a normalized coordinate in [0, 1] to a bin index in
    [0, loc_bins - 1], clamping out-of-range inputs."""
    v = min(max(float(value), 0.0), 1.0)
    return min(int(v * loc_bins), loc_bins - 1)


def loc_tokens_from_list(coords: Sequence[float], loc_bins: int) -> str:
    """Render a flat list of normalized coordinates as concatenated
    ``<loc_i>`` tokens, e.g. ``[0.1, 0.2] -> "<loc_100><loc_200>"``."""
    return "".join(f"<loc_{quantize_coord(c, loc_bins)}>" for c in coords)
