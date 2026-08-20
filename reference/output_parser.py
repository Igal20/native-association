"""Parse / validate the sports-unified generated string.

Inverse of ``serializer.py``: turns a decoded
(``skip_special_tokens=False``) target string back into a structured dict with
the same shape as the ground-truth ``players`` list, so the eval metrics can
compare PRED vs GT symmetrically.

Loc bins invert via ``coord = bin / (LOC_BINS - 1)`` (matching the ``to_bin``
rounding in ``serializer.py``).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from grammar import LOC_BINS

from grammar import (
    FLAG_TO_GT_TEAM,
    TEAM_A,
    TEAM_B,
    TEAM_OTHER,
    UNKNOWN,
)

_LOC_RE = re.compile(r"<loc_(\d+)>")
_PLAYER_RE = re.compile(r"<player>(.*?)</player>", re.DOTALL)
_BBOX_RE = re.compile(r"<bbox>(.*?)</bbox>", re.DOTALL)
_TEAM_RE = re.compile(r"<team>(<team_a>|<team_b>|<team_other>|<unknown>)")
_OCR_RE = re.compile(r"<ocr>(.*?)((?:<loc_\d+>){8})", re.DOTALL)
# 8-loc quad anywhere in an OCR segment (after_text mode). Absence => no-quad
# (ocr_quads="none", E1b) item whose text runs to the next <ocr>/end of block.
_OCR_QUAD8_RE = re.compile(r"(?:<loc_\d+>){8}")
_SPORT_RE = re.compile(r"<sport>(.*?)(?=<stype>|<player>|<gdesc>|</s>|$)", re.DOTALL)
_STYPE_RE = re.compile(r"<stype>(.*?)(?=<player>|<gdesc>|</s>|$)", re.DOTALL)
_GDESC_RE = re.compile(r"<gdesc>(.*?)(?=</s>|$)", re.DOTALL)

_TEAM_FLAGS = (TEAM_A, TEAM_B, TEAM_OTHER, UNKNOWN)


def _bin_to_coord(b: int) -> float:
    denom = max(1, LOC_BINS - 1)
    return max(0.0, min(1.0, b / denom))


def _clean_special(text: str) -> str:
    """Drop any stray special-token markers and whitespace from a free-text span."""
    text = re.sub(r"<s>|</s>|<pad>", "", text)
    return text.strip()


def _parse_bbox(inner: str) -> Optional[List[float]]:
    bins = [int(m) for m in _LOC_RE.findall(inner)]
    if len(bins) < 4:
        return None
    x1, y1, x2, y2 = (_bin_to_coord(b) for b in bins[:4])
    return [x1, y1, x2, y2]


def _parse_ocr_items(block: str) -> List[Dict[str, Any]]:
    """Parse a player block's OCR items, mode-agnostically.

    Handles BOTH schema variants without needing the ``ocr_quads`` flag:
      * after_text (E1): ``<ocr>TEXT<loc*8>`` -> ``{text, ocr_quad}``.
      * none      (E1b): ``<ocr>TEXT`` with no locs -> ``{text, ocr_quad: None}``;
        the text runs to the next ``<ocr>`` or the end of the block.
    Detection is per item: a segment carrying an 8-loc run is after_text, else
    no-quad (the OCR-text alphabet excludes loc ids, so a segment has either 0
    or exactly 8 locs).
    """
    items: List[Dict[str, Any]] = []
    # Split on <ocr>; chunk[0] is the pre-OCR content (bbox/team), skip it.
    for seg in block.split("<ocr>")[1:]:
        quad_m = _OCR_QUAD8_RE.search(seg)
        if quad_m:
            text = _clean_special(seg[: quad_m.start()])
            bins = [int(x) for x in _LOC_RE.findall(quad_m.group(0))][:8]
            quad = [[_bin_to_coord(bins[2 * i]), _bin_to_coord(bins[2 * i + 1])] for i in range(4)]
            items.append({"text": text, "ocr_quad": quad})
        else:
            text = _clean_special(seg)
            items.append({"text": text, "ocr_quad": None})
    return items


def _primary_jersey(ocr_items: List[Dict[str, Any]]) -> str:
    """First digit-only OCR text = the primary jersey number (GT convention).

    jersey_number is digit-only by definition on BOTH GT and pred; players
    without a digit-only OCR item are excluded from jersey metrics (they still
    count in multi_ocr_f1 and detection).
    """
    for it in ocr_items:
        t = (it.get("text") or "").strip()
        if t.isdigit():
            return t
    return ""


def parse_sports_output(text: str) -> Dict[str, Any]:
    """Parse a decoded target string into ``{sport, scene_type, players, ...}``."""
    text = text or ""

    sport_m = _SPORT_RE.search(text)
    stype_m = _STYPE_RE.search(text)
    gdesc_m = _GDESC_RE.search(text)

    players: List[Dict[str, Any]] = []
    for pm in _PLAYER_RE.finditer(text):
        block = pm.group(1)
        bbox = None
        bbox_m = _BBOX_RE.search(block)
        if bbox_m:
            bbox = _parse_bbox(bbox_m.group(1))
        if bbox is None:
            continue
        team_m = _TEAM_RE.search(block)
        team_flag = team_m.group(1) if team_m else UNKNOWN
        ocr_items = _parse_ocr_items(block)
        players.append(
            {
                "bbox": bbox,
                "team_flag": team_flag,
                "team": FLAG_TO_GT_TEAM.get(team_flag, "Unknown"),
                "jersey_number": _primary_jersey(ocr_items),
                "ocr_items": ocr_items,
            }
        )

    return {
        "sport_type": _clean_special(sport_m.group(1)) if sport_m else "",
        "scene_type": _clean_special(stype_m.group(1)) if stype_m else "",
        "scene_description": _clean_special(gdesc_m.group(1)) if gdesc_m else "",
        "players": players,
        "num_players": len(players),
    }


def validate_sports_output(text: str) -> Dict[str, bool]:
    """Cheap structural checks (grammar-constrained decode should give 1.0)."""
    text = text or ""
    n_open = text.count("<player>")
    n_close = text.count("</player>")
    parsed = parse_sports_output(text)
    checks = {
        "has_sport": "<sport>" in text,
        "has_stype": "<stype>" in text,
        "has_gdesc": "<gdesc>" in text,
        "has_eos": "</s>" in text,
        "players_balanced": n_open == n_close,
        "all_players_have_bbox": all(p.get("bbox") for p in parsed["players"]),
        "all_players_have_team": all(
            p.get("team_flag") in _TEAM_FLAGS for p in parsed["players"]
        ),
    }
    checks["overall_compliant"] = all(checks.values())
    return checks


def schema_compliance_rate(texts: List[str]) -> Dict[str, Any]:
    """Aggregate ``overall_compliant`` across a list of decoded strings."""
    if not texts:
        return {"compliance_rate": 0.0, "n": 0, "n_compliant": 0}
    flags = [validate_sports_output(t)["overall_compliant"] for t in texts]
    n_ok = sum(1 for f in flags if f)
    return {"compliance_rate": n_ok / len(texts), "n": len(texts), "n_compliant": n_ok}


__all__ = [
    "parse_sports_output",
    "validate_sports_output",
    "schema_compliance_rate",
]
