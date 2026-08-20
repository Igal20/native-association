"""EvalFrame -> target string. **The schema is LOCKED in this module.**

Every consumer (E0 report / budget audit / eyeball smoke AND E1's training
dataset) imports :func:`serialize_frame` from here so the target string is
produced in exactly one place. Nothing is hand-copied.

Schema v1 (decision 2026-07-05) -- NO ``<nath>``, NO ``<facing>``.

``<SPORTS_UNIFIED>`` is the ENCODER task prompt (WIDER convention), NOT part of
the decoder target -- the target begins at ``<sport>`` (epoch-1 PRED is
expected to start ``</s><s><sport>``). The decoder target string is::

    <sport>{X}<stype>{Y}
      <player><bbox><loc*4></bbox><team>{<team_a>|<team_b>|<team_other>|<unknown>}
          <ocr>{TEXT}<loc*8> ...(0..K OCR items, largest quad first)...
      </player>...
    <gdesc>{scene_description}</s>

Rules (WIDER lessons, not negotiable):

  * fixed-slot categorical ``<team>`` -- always exactly one flag token, dirty
    values -> ``<unknown>`` (see :func:`sports_tokens.team_flag_from_gt`).
  * OCR items are variable-length, **skip-if-absent**, sorted **largest quad
    first**, and capped at ``max_ocr_per_player``.
  * ``ocr_quads`` -- quad-supervision mode (the E1<->E1b ablation axis):
      - ``"after_text"`` (E1 default): ``<ocr>TEXT<loc*8>`` (current behaviour).
      - ``"none"`` (E1b): ``<ocr>TEXT`` only -- no loc tokens. The OCR item SET
        (validity, largest-quad-first sort, cap, clean policy) is UNCHANGED; only
        the emitted loc tokens are dropped.
      - ``"before_text"`` (E4, reserved): raises ``NotImplementedError`` for now.
  * ``qmark_policy`` -- ``'?'`` is a single UNRESOLVED-glyph marker, not a
    bad-detection flag (see ``core/ocr_text.normalize_ocr_text``):
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

Sport & scene values are canonicalised via ``core.sport_vocab`` (free text
after their opener, closed by canonicalisation).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .coordinate_utils import loc_tokens_from_list
from .constants import LOC_BINS

from .sport_vocab import (
    normalize_scene_label,
    normalize_sport_label,
)
from .ocr_text import normalize_ocr_text
from .sports_tokens import (
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
# Caps -- populated from e0_out/schema_caps.json by budget_audit; ``None``
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
    # Re-exported from core so existing ``from serializer import ...`` sites keep
    # working; the canonical home is ``core/ocr_text.py``.
    "normalize_ocr_text",
]
