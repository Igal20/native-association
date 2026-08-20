"""Public constants for the Native Association release.

In the internal WSC codebase these live in a shared production module
(``implementations.v2.constants``) so the research and production
canonicalization paths cannot diverge. For the public release they are
inlined here with identical values; every value below is stated in the
paper or its supplementary material.
"""

from __future__ import annotations

from typing import Dict, FrozenSet

# Coordinate quantization: continuous [0, 1] coordinates are quantized to
# LOC_BINS location bins and emitted as <loc_0> .. <loc_{LOC_BINS-1}> tokens
# (paper Sec. 3.1).
LOC_BINS: int = 1000

# Major-sports-only taxonomy (paper Sec. 4; supplementary Tab. S13).
# Sparse residual sports fold to "Other" via normalize_sport_label.
VALID_SPORT_TYPES: FrozenSet[str] = frozenset(
    {"Soccer", "Basketball", "Football", "Hockey", "Tennis", "Other"}
)

# Variant labels merged onto a kept major BEFORE the case-fold lookup.
# Keys are case-folded (see sport_vocab.normalize_sport_label).
SPORT_LABEL_ALIASES: Dict[str, str] = {
    "ice hockey": "Hockey",
    "american football": "Football",
}

# Scene vocabulary (verbatim from the released annotation prompt,
# supplementary Sec. 5).
VALID_SCENE_TYPES: FrozenSet[str] = frozenset(
    {
        "In-Game",
        "Warm-ups",
        "Player Arrivals",
        "Press Conference",
        "Interview",
        "Locker Room",
        "Winning Ceremony",
        "Other",
    }
)
