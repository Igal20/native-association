"""Canonical OCR-text normalization for the sports-unified schema.

Lives in ``core/`` on purpose: it is consumed by three layers that MUST agree
on what "the OCR text" is --

  * the serializer  (``experiments/e0_data_prep/serializer.py``) -> training target,
  * the metric layer (``eval/metrics.py`` + ``experiments/e1_baseline_dropq/eval_sports.py``),
  * future split rebuilds (``data/build_eval_splits.py``).

``core`` is the only layer all three may import without creating an upward
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

from __future__ import annotations

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
