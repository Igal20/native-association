"""Sports-unified schema vocabulary -- the single source of truth.

This module is the sports-track analogue of
``wider_attribute_dataset/training/wider_tokens.py``. Every other module in
the USP-c3 track (E0 serializer / token_table / budget_audit, E1 grammar /
model_utils / trainer) imports the token strings from HERE so the schema is
defined exactly once.

Target string shape (schema v1, decision 2026-07-05)::

    <SPORTS_UNIFIED><sport>{X}<stype>{Y}
      <player><bbox><loc*4></bbox><team>{<team_a>|<team_b>|<team_other>|<unknown>}
          <ocr>{TEXT}<loc*8> ...(0..K, largest quad first)...
      </player>...
    <gdesc>{scene_description}</s>

Design rules carried over from the WIDER post-mortem (non-negotiable):

  * Structural + flag tokens are **dedicated single tokens** registered on
    the tokenizer (closed vocab). Sport / scene VALUES stay *free text* after
    their opener (they are closed by ``sport_vocab`` canonicalisation, which
    is cheap in BPE), and the ``<gdesc>`` body is free text. Only the tokens
    listed in :data:`SCHEMA_TOKENS` + the task marker are added to the vocab.
  * ``<loc_0>..<loc_999>`` are **reused** from the base Florence-2 vocab and
    are NOT registered here.
  * The ``<team>`` value is a dedicated flag token (``<team_a>`` etc.), never
    the free-text ``team_a`` string the cycle-2 dataset used, so it can never
    collide with player names inside ``<ocr>`` spans.

Token accounting (the "19-of-21 lesson" -- count and log the EXACT list):

  * :data:`TASK_TOKEN` = ``<SPORTS_UNIFIED>``                       -> 1 token
  * :data:`SCHEMA_TOKENS` (structural + flag)                       -> 13 tokens
  * ------------------------------------------------------------------------
  * :data:`SPORTS_CUSTOM_TOKENS` (everything registered)            -> 14 tokens

  The task marker is intentionally left UNANCHORED by
  :func:`sports_model_utils.semantic_anchor_init` (it has no natural-language
  gloss), mirroring WIDER's "20/21 seeded, 1 task token unanchored" pattern.
  So expect ``seeded 13/14`` in the anchor-init log.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .constants import LOC_BINS


# =========================================================================
# Task marker (encoder prompt) -- registered but NOT semantically anchored
# =========================================================================

TASK_TOKEN: str = "<SPORTS_UNIFIED>"

# Encoder-side prompt expansion for the task marker. Florence-2's processor
# (`_construct_prompts`) replaces the bare ``TASK_TOKEN`` with this string, so
# the encoder always sees the same instruction. Keep the token as the PREFIX
# (WIDER convention) so the registered marker is present in the encoder input,
# then a short natural-language description of the target grammar.
#
# This MUST be registered identically at train AND eval time (it is runtime-only
# and NOT persisted by ``save_pretrained``) -- pass it via
# ``load_sports_model_and_processor(task_prompt_expansion=TASK_PROMPT)`` in both
# train_e1 and eval_sports, or the encoder input silently diverges.
TASK_PROMPT: str = (
    f"{TASK_TOKEN} Describe the sports scene. Emit the sport after <sport> and "
    f"the scene type after <stype>; for every player emit a <player> block with "
    f"<bbox><loc><loc><loc><loc></bbox>, one team flag after <team> "
    f"(<team_a>/<team_b>/<team_other>/<unknown>), then zero or more "
    f"<ocr>{{text}}<loc>*8 items; close each with </player>; finish with the "
    f"scene description after <gdesc> then </s>."
)


# =========================================================================
# Structural tokens (openers / closers)
# =========================================================================

SPORT_OPEN: str = "<sport>"          # opener; free-text canonical sport value follows
STYPE_OPEN: str = "<stype>"          # opener; free-text canonical scene value follows
GDESC_OPEN: str = "<gdesc>"          # opener; free-text scene description follows
PLAYER_OPEN: str = "<player>"        # shared (un-indexed) player block open
PLAYER_CLOSE: str = "</player>"      # shared player block close
BBOX_OPEN: str = "<bbox>"            # reuse the WIDER name; begins a normalized xyxy box
BBOX_CLOSE: str = "</bbox>"          # reuse the WIDER name; ends the box (after exactly 4 locs)
TEAM_OPEN: str = "<team>"            # opener; a single team flag token follows
OCR_OPEN: str = "<ocr>"             # begins one OCR item: free text then exactly 8 locs

EOS_TOKEN: str = "</s>"              # reused from base vocab; terminates the sequence


# =========================================================================
# Team flag tokens (fixed-slot categorical value -- dedicated single tokens)
# =========================================================================

TEAM_A: str = "<team_a>"
TEAM_B: str = "<team_b>"
TEAM_OTHER: str = "<team_other>"
UNKNOWN: str = "<unknown>"           # dirty / undeterminable team affiliation

TEAM_VALUE_TOKENS: List[str] = [TEAM_A, TEAM_B, TEAM_OTHER, UNKNOWN]

# GT ("Team A" / "Team B" / "other") -> dedicated flag token. Anything not in
# this map (empty string, "Unknown", stray labels) canonicalises to <unknown>.
GT_TEAM_TO_FLAG: Dict[str, str] = {
    "Team A": TEAM_A,
    "Team B": TEAM_B,
    "other": TEAM_OTHER,
}
# Case-folded view used for the actual lookup so label-casing variants map to
# the same flag (e.g. GT carries both "other" and "Other" -> <team_other>;
# "team a" / "TEAM A" -> <team_a>). Keep GT_TEAM_TO_FLAG as the canonical
# (human-readable) source of truth above.
_GT_TEAM_TO_FLAG_FOLDED: Dict[str, str] = {
    k.casefold(): v for k, v in GT_TEAM_TO_FLAG.items()
}
# Reverse map for the parser / eval; <unknown> deliberately has no GT string.
FLAG_TO_GT_TEAM: Dict[str, str] = {
    TEAM_A: "Team A",
    TEAM_B: "Team B",
    TEAM_OTHER: "other",
    UNKNOWN: "Unknown",
}


def team_flag_from_gt(raw: Optional[str]) -> str:
    """Map a GT ``team`` label to its dedicated flag token.

    Matching is case-insensitive, so ``"other"`` and ``"Other"`` both map to
    :data:`TEAM_OTHER` (the cycle-3 GT carries both casings). Dirty / missing /
    undeterminable values (``""``, ``"Unknown"``, ``None``, anything
    unrecognised) collapse to :data:`UNKNOWN`. This is the fixed-slot
    categorical rule from the WIDER post-mortem: the slot always emits exactly
    one of the four flag tokens.
    """
    s = (raw or "").strip().casefold()
    return _GT_TEAM_TO_FLAG_FOLDED.get(s, UNKNOWN)


# =========================================================================
# The registered token set
# =========================================================================

# Structural + flag tokens (the "13"). This list is the REGISTRATION / logical
# grouping order for the token table -- it is NOT the literal emission order.
# The true emission order is defined by the serializer + the FSM grammar:
#
#   <sport>{X} <stype>{Y}
#     ( <player> <bbox> <loc>x4 </bbox> <team>{flag} (<ocr>{text}<loc>x8)* </player> )*
#   <gdesc>{desc} </s>
#
# In particular <gdesc> is emitted LAST (before </s>), not third; and <team>
# is a fixed slot always followed by EXACTLY ONE of
# {<team_a>, <team_b>, <team_other>, <unknown>} (enforced in the FSM's
# _S_TEAM_VALUE state and by team_flag_from_gt in the serializer).
SCHEMA_TOKENS: List[str] = [
    SPORT_OPEN,      # header opener 1
    STYPE_OPEN,      # header opener 2
    GDESC_OPEN,      # emitted LAST (before </s>), listed here for grouping
    PLAYER_OPEN,     # per-player block open
    PLAYER_CLOSE,    # per-player block close
    BBOX_OPEN,       # bbox open (then exactly 4 <loc_>)
    BBOX_CLOSE,      # bbox close
    TEAM_OPEN,       # team slot open -> followed by EXACTLY ONE flag below
    TEAM_A,          # team flag value
    TEAM_B,          # team flag value
    TEAM_OTHER,      # team flag value
    UNKNOWN,         # team flag value (dirty / undeterminable)
    OCR_OPEN,        # 0..K per player: <ocr>{text}<loc>x8
]

# Everything we add to the tokenizer: the task marker first, then the schema
# tokens. ``<loc_*>`` and ``</s>`` / ``<s>`` are reused from base Florence-2.
SPORTS_CUSTOM_TOKENS: List[str] = [TASK_TOKEN] + SCHEMA_TOKENS

EXPECTED_NUM_SCHEMA_TOKENS: int = 13
EXPECTED_NUM_CUSTOM_TOKENS: int = 14


# =========================================================================
# Semantic anchor phrases (leading space added at tokenize time)
# =========================================================================
#
# Each new token row is seeded with the mean sub-word embedding of its gloss
# so the decoder starts from a sensible neighbourhood instead of a random
# row. TASK_TOKEN has no natural gloss and is intentionally omitted (left at
# the resize_token_embeddings default), so anchor init reports "13/14".

ANCHOR_PHRASES: Dict[str, str] = {
    SPORT_OPEN:   "sport",
    STYPE_OPEN:   "scene type setting",
    GDESC_OPEN:   "description caption scene",
    PLAYER_OPEN:  "player person athlete",
    PLAYER_CLOSE: "player person athlete",
    BBOX_OPEN:    "box region",
    BBOX_CLOSE:   "box region",
    TEAM_OPEN:    "team side",
    TEAM_A:       "team a home",
    TEAM_B:       "team b away",
    TEAM_OTHER:   "other neutral referee official",
    UNKNOWN:      "unknown unclear unsure",
    OCR_OPEN:     "text jersey number writing",
}


def anchor_phrase_for(token: str) -> Optional[str]:
    """Return the anchor gloss for ``token`` or ``None`` (task marker / unknown)."""
    return ANCHOR_PHRASES.get(token)


# =========================================================================
# Structural-token set for the constrained decoder's free-text spans
# =========================================================================
#
# Free-text spans (sport / scene values, <gdesc> body, <ocr> text) are decoded
# as "any vocab id EXCEPT the structural/flag tokens and BOS", so the model can
# copy arbitrary text but can never emit a structural token mid-span. The
# terminator that closes each free-text span (e.g. <stype> after <sport>value,
# <loc_> after <ocr>text) is added back to the allow-set by the grammar.

STRUCTURAL_TOKENS: List[str] = list(SPORTS_CUSTOM_TOKENS)


def token_map() -> Dict[str, object]:
    """A JSON-serialisable summary of the vocabulary for logging / assertions."""
    return {
        "task_token": TASK_TOKEN,
        "schema_tokens": list(SCHEMA_TOKENS),
        "team_value_tokens": list(TEAM_VALUE_TOKENS),
        "custom_tokens": list(SPORTS_CUSTOM_TOKENS),
        "num_schema_tokens": len(SCHEMA_TOKENS),
        "num_custom_tokens": len(SPORTS_CUSTOM_TOKENS),
        "expected_num_schema_tokens": EXPECTED_NUM_SCHEMA_TOKENS,
        "expected_num_custom_tokens": EXPECTED_NUM_CUSTOM_TOKENS,
        "anchored_tokens": sorted(ANCHOR_PHRASES.keys()),
        "unanchored_tokens": [
            t for t in SPORTS_CUSTOM_TOKENS if t not in ANCHOR_PHRASES
        ],
        "loc_bins": LOC_BINS,
        "gt_team_to_flag": dict(GT_TEAM_TO_FLAG),
        "reuses_from_base_vocab": ["<s>", "</s>", "<pad>", f"<loc_0>..<loc_{LOC_BINS - 1}>"],
    }


__all__ = [
    "TASK_TOKEN",
    "TASK_PROMPT",
    "SPORT_OPEN",
    "STYPE_OPEN",
    "GDESC_OPEN",
    "PLAYER_OPEN",
    "PLAYER_CLOSE",
    "BBOX_OPEN",
    "BBOX_CLOSE",
    "TEAM_OPEN",
    "OCR_OPEN",
    "EOS_TOKEN",
    "TEAM_A",
    "TEAM_B",
    "TEAM_OTHER",
    "UNKNOWN",
    "TEAM_VALUE_TOKENS",
    "GT_TEAM_TO_FLAG",
    "FLAG_TO_GT_TEAM",
    "team_flag_from_gt",
    "SCHEMA_TOKENS",
    "SPORTS_CUSTOM_TOKENS",
    "STRUCTURAL_TOKENS",
    "EXPECTED_NUM_SCHEMA_TOKENS",
    "EXPECTED_NUM_CUSTOM_TOKENS",
    "ANCHOR_PHRASES",
    "anchor_phrase_for",
    "token_map",
]
