"""EvalFrame -> target string. **The schema is LOCKED in this module.**

Every consumer (dataset audits, smoke tests, AND the training dataset)
imports :func:`serialize_frame` from here so the target string is produced
in exactly one place. Nothing is hand-copied. Keep this property in any port.

Schema v1 -- deliberately minimal (see the paper for what was dropped).

``<SPORTS_UNIFIED>`` is the ENCODER task prompt (WIDER convention), NOT part of
the decoder target -- the target begins at ``<sport>`` (epoch-1 PRED is
expected to start ``</s><s><sport>``). The decoder target string is::

    <sport>{X}<stype>{Y}
      <player><bbox><loc*4></bbox><team>{<team_a>|<team_b>|<team_other>|<unknown>}
          <ocr>{TEXT}<loc*8> ...(0..K OCR items, largest quad first)...
      </player>...
    <gdesc>{scene_description}</s>

Rules (lessons from the WIDER-Attribute port, not negotiable):

  * fixed-slot categorical ``<team>`` -- always exactly one flag token, dirty
    values -> ``<unknown>`` (see :func:`grammar.team_flag_from_gt`).
  * OCR items are variable-length, **skip-if-absent**, sorted **largest quad
    first**, and capped at ``max_ocr_per_player``.
  * ``ocr_quads`` -- quad-supervision mode (the E1<->E1b ablation axis):
      - ``"after_text"`` (E1 default): ``<ocr>TEXT<loc*8>`` (current behaviour).
      - ``"none"`` (E1b): ``<ocr>TEXT`` only -- no loc tokens. The OCR item SET
        (validity, largest-quad-first sort, cap, clean policy) is UNCHANGED; only
        the emitted loc tokens are dropped.
      - ``"before_text"`` (E4, reserved): raises ``NotImplementedError`` for now.
  * ``qmark_policy`` -- ``'?'`` is a single UNRESOLVED-glyph marker, not a
    bad-detection flag (see ``normalize_ocr_text`` below):
      - ``"clean"`` (E1 default): repair -- strip boundary ``'?'`` runs, turn
        interior ``'?'`` runs into a space, drop only all-``'?'`` items.
      - ``"drop"``  (E1c ablation): remove the whole OCR item if it contains ``'?'``.
      - ``"keep"``  (E1b ablation): pass the text verbatim, ``'?'`` included.
    Text is otherwise verbatim; quads are NEVER modified by any policy.
  * quads = ``ocr_items[].ocr_quad`` flattened to 8 loc tokens.
  * players sorted by bbox area (desc), capped at ``max_players``. All players
    share the same ``<player>`` / ``</player>`` tags (the per-frame cap lives
    in the decode grammar, not in indexed tags).
  * loc quantization via the shared ``loc_tokens_from_list``.
  * v1 DROPS ``num_athletes`` / ``facing`` / ``action_name`` /
    ``overall_sentiment`` / ``main_characters`` / ``scene_name`` -- they stay
    in the GT dict for later ablations; the serializer simply does not emit
    them.

Sport & scene values are canonicalised by the helpers below (free text
after their opener, closed by canonicalisation).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


from grammar import LOC_BINS


from grammar import (
    BBOX_CLOSE,
    BBOX_OPEN,
    EOS_TOKEN,
    GDESC_OPEN,
    OCR_OPEN,
    PLAYER_CLOSE,
    PLAYER_OPEN,
    SPORT_OPEN,
    STYPE_OPEN,
    TEAM_OPEN,
    team_flag_from_gt,
)
# Public release: EvalFrame is inlined here (internally it lives in the
# dataset-split I/O module). Only the fields used by serialization are needed.
from dataclasses import dataclass as _dataclass
from typing import Any as _Any, Dict as _Dict


# ======================================================================
# Inlined helpers (vocab canonicalization, OCR-text policy, quantization)
# ======================================================================
from grammar import SPORT_LABEL_ALIASES, VALID_SPORT_TYPES, VALID_SCENE_TYPES


# ======================================================================
"""Sport-type vocabulary helpers.

Single source of truth re-exported from ``grammar.py``, plus a
case-insensitive normalizer that maps unknown labels to ``"Other"``.
"""


from typing import Dict, Optional



# SPORT_LABEL_ALIASES is defined ONCE in grammar.py (the single source of
# truth) and imported here so canonicalization paths can never diverge. It maps
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
    well-formed ``<sport>{label}`` opener, even on legacy frames whose
    GT didn't carry ``sport_type``.
    """
    s = (raw or "Other").strip()
    folded = s.casefold()
    if folded in SPORT_LABEL_ALIASES:
        return SPORT_LABEL_ALIASES[folded]
    case_fold = {x.casefold(): x for x in VALID_SPORT_TYPES}
    return case_fold.get(folded, "Other")


# =========================================================================
# Scene-type vocabulary (sport & scene values canonicalized here, in the
# same module, so serialization and vocabulary can never drift apart)
# =========================================================================

# Aliases applied BEFORE the case-fold lookup against VALID_SCENE_TYPES. Keys are
# case-folded. These recover GT labels that are semantically one of the
# VALID_SCENE_TYPES but whose raw string doesn't match (spacing/synonym), so they
# are NOT silently dumped into "Other" (a dataset-audit finding: audit your
# label tails before serializing, or aliasable labels bleed into "Other"). Anything
# still unmatched (Player Shots, Non-In-Game, Post-Game, Handshake, Team Huddle,
# Coin Toss, Stadium Shot, ...) falls through to "Other" via the fallback below.
SCENE_LABEL_ALIASES: Dict[str, str] = {
    # "Warm Ups" (space) is the same class as canonical "Warm-ups" (hyphen).
    # ~281 frames in our corpus -- the single biggest label lost to the mismatch.
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
# Sport merge mapping -- APPLIED (major-sports-only taxonomy).
# VALID_SPORT_TYPES = {Soccer, Basketball, Football, Hockey, Tennis, Other};
# the merges live in SPORT_LABEL_ALIASES (defined in grammar.py, the single
# source of truth) and the rest fold to "Other".
# =========================================================================

SPORT_MERGE_PROPOSAL: Dict[str, object] = {
    "applied_aliases": dict(SPORT_LABEL_ALIASES),  # defined in grammar.py
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
"""Canonical OCR-text normalization for the sports-unified schema.

ONE shared definition on purpose: it is consumed by three layers that MUST agree
on what "the OCR text" is --

  * the serializer (``serializer.py``) -> the training target,
  * the metric layer (``metrics.py`` + your eval entry point),
  * any future split rebuilds.

This shared helper is the only thing all three may import without creating an upward
dependency, so the normalizer has exactly one home.

Why we repair instead of drop
=============================
``'?'`` is the annotation pipeline's marker for a single UNRESOLVED glyph -- a
character the OCR could not read confidently -- NOT a flag that the whole
detection is garbage. Dropping the whole item (the old ``qmark_policy="drop"``)
threw away ~32.5% of ALL OCR items (13,622 of 41,931), including ~3,235 that are
actually clean jersey numbers with one unreadable neighbour (``"?9"`` clearly
shows the ``9``). The ``clean`` policy repairs them:

    * boundary ``'?'`` runs -> stripped              ("?9"->"9",  "Bau?"->"Bau")
    * interior ``'?'`` runs -> a single SPACE         ("1?3"->"1 3", "TU?K"->"TU K")
      NEVER deleted: deleting fabricates a confident-but-wrong jersey ("1?3"->"13");
      a space keeps it out of the digit-only jersey metric (honest partial read).
    * all-``'?'`` / empty   -> ``""``                 (caller drops the item; 9 exist)
    * surrounding / repeated whitespace -> collapsed.

The example table in :data:`_EXPECTED` below doubles as the unit test (run this
module as ``__main__`` for the self-check).
"""


import re

_QMARK_RUN = re.compile(r"\?+")
_WS_RUN = re.compile(r"\s+")


def normalize_ocr_text(text: str) -> str:
    """``'?'`` = unresolved glyph, not a bad detection. See the module docstring.

    "?9"->"9"  "?NEY"->"NEY"  "Bau?"->"Bau"  "1?3"->"1 3"  "TU?K"->"TU K"
    "9?3?"->"9 3"  "??"->""  "Estrella Galicia 0,0"->unchanged
    """
    t = (text or "").strip().strip("?").strip()  # boundary '?' runs + whitespace
    t = _QMARK_RUN.sub(" ", t)                     # interior '?' runs -> single space
    return _WS_RUN.sub(" ", t).strip()             # collapse whitespace


# The docstring example table, machine-checkable (mirrors normalize_ocr_text's
# docstring). Extend both together.
_EXPECTED = {
    "?9": "9",
    "?NEY": "NEY",
    "Bau?": "Bau",
    "1?3": "1 3",
    "TU?K": "TU K",
    "9?3?": "9 3",
    "CC?": "CC",
    "?CM": "CM",
    "??": "",
    "": "",
    "Estrella Galicia 0,0": "Estrella Galicia 0,0",
}


def _self_check() -> bool:
    """Print + verify the example table; return True iff every case matches."""
    ok = True
    print("normalize_ocr_text unit table:")
    for raw, exp in _EXPECTED.items():
        got = normalize_ocr_text(raw)
        good = got == exp
        ok = ok and good
        print(f"  [{'ok' if good else 'FAIL'}] {raw!r:26s} -> {got!r:16s} (expected {exp!r})")
    print(f"  ALL {'PASS' if ok else 'FAIL'}")
    return ok


__all__ = ["normalize_ocr_text"]


if __name__ == "__main__":
    import sys

    sys.exit(0 if _self_check() else 1)
"""Coordinate quantization helpers (public re-implementation).

Internally these live in a shared production utility module; the public
version reproduces the exact behaviour used by the paper: normalized
[0, 1] coordinates are quantized to ``loc_bins`` bins and rendered as
``<loc_i>`` tokens (paper Sec. 3.1).
"""


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

# ======================================================================
# Serialization
# ======================================================================


@_dataclass
class EvalFrame:
    """One frame in the unified eval split (minimal public mirror)."""

    frame_id: str
    split: str
    image_path: str
    image_width: int
    image_height: int
    ground_truth: _Dict[str, _Any]

    @property
    def phash(self) -> str:
        parts = self.frame_id.split("::")
        return parts[-1] if parts else str(self.ground_truth.get("phash", ""))


# "clean" is the E1 default; "drop"/"keep" stay as the E1c/E1b ablation arms.
QMARK_POLICIES = ("clean", "drop", "keep")

# OCR quad-supervision mode (the E1 <-> E1b ablation axis):
#   "after_text"  -- E1 (default): <ocr>TEXT<loc*8>  (current, byte-identical).
#   "none"        -- E1b: <ocr>TEXT only, no loc tokens. OCR item cap / sorting /
#                    clean policy are all UNCHANGED (quads still gate validity +
#                    largest-quad-first ordering); only the emitted loc tokens go.
#   "before_text" -- E4 (reserved): <ocr><loc*8>TEXT ordering. Enum accepted but
#                    NOT implemented yet (raises NotImplementedError), so no dead
#                    emit branch exists.
OCR_QUAD_MODES = ("after_text", "none", "before_text")

# Sentence-boundary splitter for <gdesc> truncation (keeps the terminator).
_SENTENCE_RE = re.compile(r"[^.!?]*[.!?]|[^.!?]+$")


# =========================================================================
# Caps -- set from YOUR dataset audit (observed maxima, verified non-binding;
# the paper's corpus used max_players=12, max_ocr_per_player=17). ``None``
# means "no cap" (used while auditing the raw distribution).
# =========================================================================

@dataclass
class SchemaCaps:
    """Serialisation caps. ``None`` disables the corresponding cap."""

    max_players: Optional[int] = None
    max_ocr_per_player: Optional[int] = None
    max_gdesc_tokens: Optional[int] = None
    qmark_policy: str = "clean"
    ocr_quads: str = "after_text"

    def __post_init__(self) -> None:
        if self.qmark_policy not in QMARK_POLICIES:
            raise ValueError(
                f"qmark_policy must be one of {QMARK_POLICIES}; got {self.qmark_policy!r}"
            )
        if self.ocr_quads not in OCR_QUAD_MODES:
            raise ValueError(
                f"ocr_quads must be one of {OCR_QUAD_MODES}; got {self.ocr_quads!r}"
            )
        if self.ocr_quads == "before_text":
            # Enum reserved for E4; no emit/grammar branch exists yet. Fail loud
            # rather than silently behaving like a different mode.
            raise NotImplementedError(
                "ocr_quads='before_text' (E4: quad-before-text ordering) is not "
                "implemented yet; use 'after_text' (E1) or 'none' (E1b)."
            )

    @classmethod
    def from_dict(
        cls,
        d: Dict[str, Any],
        *,
        qmark_policy: Optional[str] = None,
        ocr_quads: Optional[str] = None,
    ) -> "SchemaCaps":
        return cls(
            max_players=d.get("max_players"),
            max_ocr_per_player=d.get("max_ocr_per_player"),
            max_gdesc_tokens=d.get("max_gdesc_tokens"),
            qmark_policy=qmark_policy or d.get("qmark_policy", "clean"),
            ocr_quads=ocr_quads or d.get("ocr_quads", "after_text"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_players": self.max_players,
            "max_ocr_per_player": self.max_ocr_per_player,
            "max_gdesc_tokens": self.max_gdesc_tokens,
            "qmark_policy": self.qmark_policy,
            "ocr_quads": self.ocr_quads,
        }


# =========================================================================
# Per-frame serialisation result (rich stats for the report / budget audit)
# =========================================================================

@dataclass
class SerializedFrame:
    text: str
    frame_id: str
    phash: str
    sport: str
    scene: str
    num_players_in: int = 0
    num_players_emitted: int = 0
    num_players_dropped_invalid: int = 0
    num_players_dropped_cap: int = 0
    num_ocr_in: int = 0
    num_ocr_emitted: int = 0
    num_ocr_dropped_qmark: int = 0
    num_ocr_dropped_invalid: int = 0
    num_ocr_dropped_cap: int = 0
    per_player_ocr_emitted: List[int] = field(default_factory=list)
    team_flag_counts: Dict[str, int] = field(default_factory=dict)
    gdesc_text: str = ""
    gdesc_truncated: bool = False


# =========================================================================
# Geometry helpers
# =========================================================================

def _flatten_quad(ocr_quad: Any) -> Optional[List[float]]:
    """``[[x,y] x4]`` (or a flat ``[x,y]*4``) -> flat ``[x,y,x,y,x,y,x,y]``.

    Returns ``None`` for malformed quads so the OCR item is skipped cleanly.
    """
    if not isinstance(ocr_quad, (list, tuple)) or not ocr_quad:
        return None
    # Already flat (8 numbers)?
    if all(isinstance(v, (int, float)) for v in ocr_quad):
        return [float(v) for v in ocr_quad] if len(ocr_quad) == 8 else None
    flat: List[float] = []
    for pair in ocr_quad:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            return None
        try:
            flat.append(float(pair[0]))
            flat.append(float(pair[1]))
        except (TypeError, ValueError):
            return None
    return flat if len(flat) == 8 else None


def _polygon_area(flat_quad: Sequence[float]) -> float:
    """Shoelace area of a 4-point polygon given as ``[x,y]*4`` (normalized)."""
    xs = flat_quad[0::2]
    ys = flat_quad[1::2]
    n = len(xs)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += xs[i] * ys[j] - xs[j] * ys[i]
    return abs(area) / 2.0


def _bbox_area(bbox: Sequence[float]) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _valid_bbox(bbox: Any) -> Optional[Tuple[float, float, float, float]]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return None
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        return None
    return x1, y1, x2, y2


# =========================================================================
# <gdesc> sentence-boundary truncation (token budget)
# =========================================================================

def _sentences(text: str) -> List[str]:
    return [m.group(0).strip() for m in _SENTENCE_RE.finditer(text) if m.group(0).strip()]


def _truncate_gdesc(
    text: str,
    max_tokens: Optional[int],
    tokenizer: Any,
) -> Tuple[str, bool]:
    """Truncate ``text`` to <= ``max_tokens`` at a sentence boundary.

    Requires a tokenizer to count tokens. If none is provided (or no cap),
    returns the text unchanged. If even the first sentence overflows, hard
    truncates by tokens and decodes.
    """
    if not text or max_tokens is None or tokenizer is None:
        return text, False

    def n_tok(s: str) -> int:
        return len(tokenizer(s, add_special_tokens=False).input_ids)

    if n_tok(text) <= max_tokens:
        return text, False

    kept: List[str] = []
    for sent in _sentences(text):
        candidate = (" ".join(kept + [sent])).strip()
        if n_tok(candidate) <= max_tokens:
            kept.append(sent)
        else:
            break
    if kept:
        return " ".join(kept).strip(), True

    # First sentence already overflows -> hard token truncation.
    ids = tokenizer(text, add_special_tokens=False).input_ids[:max_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True).strip(), True


# =========================================================================
# The serializer
# =========================================================================

def _iter_ocr_items(player: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = player.get("ocr_items")
    return items if isinstance(items, list) else []


def _clean_ocr(text: str, qmark_policy: str) -> Tuple[Optional[str], bool]:
    """Apply the ``qmark_policy`` to one OCR text.

    Returns ``(text_or_None, dropped_for_qmark)``.
      * ``clean`` -- repair via :func:`normalize_ocr_text`; the item is dropped
        only if it was ALL ``'?'`` (normalizes to ``""``).
      * ``drop``  -- remove the whole item when its text contains ``'?'``.
      * ``keep``  -- pass the text verbatim (``'?'`` included).
    """
    t = (text or "").strip()
    if not t:
        return None, False
    if qmark_policy == "clean":
        cleaned = normalize_ocr_text(t)
        # dropped_for_qmark := became empty (all-'?'); ~9 items dataset-wide.
        return (cleaned or None), (cleaned == "")
    if qmark_policy == "drop" and "?" in t:
        return None, True
    return t, False


def serialize_frame(
    frame: EvalFrame,
    caps: SchemaCaps,
    *,
    tokenizer: Any = None,
) -> SerializedFrame:
    """Serialise one :class:`EvalFrame` into the locked target string.

    Args:
        frame: an EvalFrame from ``eval_split_io.load_eval_split``.
        caps: serialisation caps + qmark policy.
        tokenizer: optional HF tokenizer, only needed when
            ``caps.max_gdesc_tokens`` is set (for sentence-boundary truncation).
    """
    gt: Dict[str, Any] = frame.ground_truth or {}
    sport = normalize_sport_label(gt.get("sport_type"))
    scene = normalize_scene_label(gt.get("scene_type"))

    raw_players = gt.get("players") or []
    result = SerializedFrame(
        text="",
        frame_id=frame.frame_id,
        phash=frame.phash,
        sport=sport,
        scene=scene,
        num_players_in=len(raw_players),
    )

    # --- Collect valid players with their bbox area (for area-desc sort) ----
    valid_players: List[Tuple[float, Tuple[float, float, float, float], Dict[str, Any]]] = []
    for p in raw_players:
        if not isinstance(p, dict):
            result.num_players_dropped_invalid += 1
            continue
        bbox = _valid_bbox(p.get("bbox"))
        if bbox is None:
            result.num_players_dropped_invalid += 1
            continue
        valid_players.append((_bbox_area(bbox), bbox, p))

    valid_players.sort(key=lambda t: t[0], reverse=True)
    if caps.max_players is not None and len(valid_players) > caps.max_players:
        result.num_players_dropped_cap = len(valid_players) - caps.max_players
        valid_players = valid_players[: caps.max_players]

    # --- Header (NB: <SPORTS_UNIFIED> is the encoder prompt, NOT emitted) ---
    parts: List[str] = [f"{SPORT_OPEN}{sport}", f"{STYPE_OPEN}{scene}"]

    # --- Player blocks -----------------------------------------------------
    for _area, bbox, p in valid_players:
        parts.append(PLAYER_OPEN)
        parts.append(
            BBOX_OPEN + loc_tokens_from_list(list(bbox), LOC_BINS) + BBOX_CLOSE
        )
        team_flag = team_flag_from_gt(p.get("team"))
        result.team_flag_counts[team_flag] = result.team_flag_counts.get(team_flag, 0) + 1
        parts.append(f"{TEAM_OPEN}{team_flag}")

        ocr_here = _serialize_player_ocr(p, caps, result)
        parts.extend(ocr_here)
        result.per_player_ocr_emitted.append(len(ocr_here))

        parts.append(PLAYER_CLOSE)
        result.num_players_emitted += 1

    # --- gdesc + EOS -------------------------------------------------------
    raw_desc = (gt.get("scene_description") or gt.get("description") or "").strip()
    gdesc, truncated = _truncate_gdesc(raw_desc, caps.max_gdesc_tokens, tokenizer)
    result.gdesc_text = gdesc
    result.gdesc_truncated = truncated
    parts.append(f"{GDESC_OPEN}{gdesc}")
    parts.append(EOS_TOKEN)

    result.text = "".join(parts)
    return result


def _serialize_player_ocr(
    player: Dict[str, Any],
    caps: SchemaCaps,
    result: SerializedFrame,
) -> List[str]:
    """Build the OCR fragment list for one player (area-desc, capped, policy)."""
    scored: List[Tuple[float, str, List[float]]] = []
    for oi in _iter_ocr_items(player):
        if not isinstance(oi, dict):
            continue
        result.num_ocr_in += 1
        quad = _flatten_quad(oi.get("ocr_quad"))
        cleaned, dropped_q = _clean_ocr(oi.get("text", ""), caps.qmark_policy)
        if dropped_q:
            result.num_ocr_dropped_qmark += 1
            continue
        if cleaned is None or quad is None:
            result.num_ocr_dropped_invalid += 1
            continue
        scored.append((_polygon_area(quad), cleaned, quad))

    scored.sort(key=lambda t: t[0], reverse=True)
    if caps.max_ocr_per_player is not None and len(scored) > caps.max_ocr_per_player:
        result.num_ocr_dropped_cap += len(scored) - caps.max_ocr_per_player
        scored = scored[: caps.max_ocr_per_player]

    fragments: List[str] = []
    emit_quads = caps.ocr_quads == "after_text"
    for _area, text, quad in scored:
        if emit_quads:
            fragments.append(f"{OCR_OPEN}{text}{loc_tokens_from_list(quad, LOC_BINS)}")
        else:  # "none" (E1b): text only, no loc tokens. Validity/sort/cap above
            # still used the quad, so the item SET is identical to after_text.
            fragments.append(f"{OCR_OPEN}{text}")
        result.num_ocr_emitted += 1
    return fragments


__all__ = [
    "QMARK_POLICIES",
    "OCR_QUAD_MODES",
    "SchemaCaps",
    "SerializedFrame",
    "serialize_frame",
    # Canonical home of OCR-text normalization in this release.
    "normalize_ocr_text",
]
