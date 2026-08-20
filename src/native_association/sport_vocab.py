"""Sport-type vocabulary helpers shared by every ECCV-2026 experiment.

Single source of truth re-exported from ``implementations.v2.constants``,
plus a case-insensitive normalizer that maps unknown labels to ``"Other"``.

Importing here (rather than directly from ``implementations.v2``) keeps the
experiment surface area decoupled from the production constants module so
we can swap it out without touching every variant runner.
"""

from __future__ import annotations

from typing import Dict, Optional

from .constants import (
    SPORT_LABEL_ALIASES,
    VALID_SCENE_TYPES,
    VALID_SPORT_TYPES,
)


# SPORT_LABEL_ALIASES is defined ONCE in implementations.v2.constants (the
# single source of truth shared with the production dataset) and imported here
# so the ECCV and production canonicalization paths can never diverge. It maps
# variant labels onto a kept major BEFORE the case-fold lookup, e.g.:
#   "Ice Hockey"        -> "Hockey"    (Hockey family)
#   "American Football" -> "Football"  (same gridiron sport, GT split the label)
# Distinct-but-sparse / zero-sample labels (Australian Rules Football, Rugby,
# Racing, Baseball) are NOT aliased; absent from VALID_SPORT_TYPES they fold to
# "Other" through the fallback in normalize_sport_label.


def normalize_sport_label(raw: Optional[str]) -> str:
    """Map an arbitrary GT sport label to one of ``VALID_SPORT_TYPES``.

    Resolution order:
      1. Aliases in ``SPORT_LABEL_ALIASES`` (case-insensitive).
      2. Case-insensitive exact match against ``VALID_SPORT_TYPES``.
      3. Fallback to ``"Other"``.

    The alias step lets us merge near-duplicate labels (e.g. "Ice Hockey"
    -> "Hockey") without bloating ``VALID_SPORT_TYPES`` -- the model only
    ever sees the canonical right-hand side, so the alias is invisible at
    the token level.

    Unknown / missing labels fall back to ``"Other"`` so we always emit a
    well-formed ``<sport>{label}`` opener, even on cycle-2 frames where
    the original schema didn't carry ``sport_type``.
    """
    s = (raw or "Other").strip()
    folded = s.casefold()
    if folded in SPORT_LABEL_ALIASES:
        return SPORT_LABEL_ALIASES[folded]
    case_fold = {x.casefold(): x for x in VALID_SPORT_TYPES}
    return case_fold.get(folded, "Other")


# =========================================================================
# Scene-type vocabulary (shares this module per the E0 pack: "Sport & scene
# values canonicalized via core/sport_vocab.py")
# =========================================================================

# Aliases applied BEFORE the case-fold lookup against VALID_SCENE_TYPES. Keys are
# case-folded. These recover cycle-3 GT labels that are semantically one of the
# VALID_SCENE_TYPES but whose raw string doesn't match (spacing/synonym), so they
# are NOT silently dumped into "Other" (E0 report 20260706 finding). Anything
# still unmatched (Player Shots, Non-In-Game, Post-Game, Handshake, Team Huddle,
# Coin Toss, Stadium Shot, ...) falls through to "Other" via the fallback below.
SCENE_LABEL_ALIASES: Dict[str, str] = {
    # "Warm Ups" (space) is the same class as canonical "Warm-ups" (hyphen).
    # ~281 cycle-3 frames -- the single biggest label lost to the mismatch.
    "warm ups": "Warm-ups",
    "warmups": "Warm-ups",
    # VALID "Player Arrivals" is defined as "athletes entering venue, tunnel
    # walks, pre-game arrivals" -- both of these raw labels belong there.
    "player entrance": "Player Arrivals",
    "tunnel walk": "Player Arrivals",
}


def normalize_scene_label(raw: Optional[str]) -> str:
    """Map an arbitrary GT scene label to one of ``VALID_SCENE_TYPES``.

    Resolution order mirrors :func:`normalize_sport_label`:
      1. Aliases in ``SCENE_LABEL_ALIASES`` (case-insensitive).
      2. Case-insensitive exact match against ``VALID_SCENE_TYPES``.
      3. Fallback to ``"Other"`` (the "scene tail -> Other" collapse).
    """
    s = (raw or "Other").strip()
    folded = s.casefold()
    if folded in SCENE_LABEL_ALIASES:
        return SCENE_LABEL_ALIASES[folded]
    case_fold = {x.casefold(): x for x in VALID_SCENE_TYPES}
    return case_fold.get(folded, "Other")


# =========================================================================
# Sport merge mapping (D2/D3) -- emitted verbatim in the E0 report. Now APPLIED
# (major-sports-only taxonomy). VALID_SPORT_TYPES = {Soccer, Basketball,
# Football, Hockey, Tennis, Other}; the merges live in SPORT_LABEL_ALIASES
# (constants.py, single source of truth) and the rest fold to "Other".
# =========================================================================

SPORT_MERGE_PROPOSAL: Dict[str, object] = {
    "applied_aliases": dict(SPORT_LABEL_ALIASES),  # live in constants.py
    "valid_sport_types": list(VALID_SPORT_TYPES),
    "decisions": {
        # Ice Hockey folded into Hockey (label variant of the same family).
        "hockey_family": {
            "merge": ["Hockey", "Ice Hockey"],
            "into": "Hockey",
            "status": "applied",
        },
        # "Football" == gridiron / American football in this dataset (GT scene
        # descriptions reference quarterbacks, linemen, NFL teams) and is DISTINCT
        # from "Soccer" (association football). The sparse "American Football"
        # label is the same sport -> merged into "Football".
        "football_family": {
            "soccer": {"label": "Soccer", "into": "Soccer", "status": "kept (major)"},
            "gridiron": {
                "merge": ["Football", "American Football"],
                "into": "Football",
                "status": "applied",
            },
        },
        # Distinct-but-sparse / zero-sample sports -> fold to "Other" (not aliased;
        # simply absent from VALID_SPORT_TYPES so the fallback catches them).
        "folded_to_other": {
            "labels": ["Australian Rules Football", "Rugby", "Racing", "Baseball"],
            "reason": "too sparse to learn/measure (<=14 train) or zero samples",
            "status": "applied via fallback",
        },
    },
}

SCENE_MERGE_PROPOSAL: Dict[str, object] = {
    "applied_aliases": dict(SCENE_LABEL_ALIASES),
    "proposed": {
        "scene_tail_to_other": {
            "rule": "Any scene label not in VALID_SCENE_TYPES -> 'Other'.",
            "valid_scene_types": list(VALID_SCENE_TYPES),
            "status": "applied via fallback",
        },
    },
}


__all__ = [
    "VALID_SPORT_TYPES",
    "VALID_SCENE_TYPES",
    "SPORT_LABEL_ALIASES",
    "SCENE_LABEL_ALIASES",
    "normalize_sport_label",
    "normalize_scene_label",
    "SPORT_MERGE_PROPOSAL",
    "SCENE_MERGE_PROPOSAL",
]
