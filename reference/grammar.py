"""The grammar: structural tokens + finite-state constrained decoding.

Self-contained reference implementation (paper Sec. 3.1). Merged from the
internal modules constants / sports_tokens / sports_constrained_decoding.
"""
from __future__ import annotations

"""Public constants for the Native Association release.

In the internal WSC codebase these live in a shared production module so
the research and production canonicalization paths cannot diverge. For the public release they are
inlined here with identical values; every value below is stated in the
paper or its supplementary material.
"""


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
# Keys are case-folded (see normalize_sport_label in serializer.py).
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

# ======================================================================
# Structural tokens
# ======================================================================
"""Sports-unified schema vocabulary -- the single source of truth.

Every other module (serializer, parser, constrained decoding, training)
imports the token strings from HERE so the schema is defined exactly once.
Keep that property in any port: one vocabulary module, everything else
imports from it.

Target string shape (schema v1)::

    <SPORTS_UNIFIED><sport>{X}<stype>{Y}
      <player><bbox><loc*4></bbox><team>{<team_a>|<team_b>|<team_other>|<unknown>}
          <ocr>{TEXT}<loc*8> ...(0..K, largest quad first)...
      </player>...
    <gdesc>{scene_description}</s>

Design rules carried over from the WIDER-Attribute port (non-negotiable):

  * Structural + flag tokens are **dedicated single tokens** registered on
    the tokenizer (closed vocab). Sport / scene VALUES stay *free text* after
    their opener (they are closed by vocabulary canonicalisation in serializer.py, which
    is cheap in BPE), and the ``<gdesc>`` body is free text. Only the tokens
    listed in :data:`SCHEMA_TOKENS` + the task marker are added to the vocab.
  * ``<loc_0>..<loc_999>`` are **reused** from the base Florence-2 vocab and
    are NOT registered here.
  * The ``<team>`` value is a dedicated flag token (``<team_a>`` etc.), never
    a free-text ``team_a`` string, so it can never collide with player names
    inside ``<ocr>`` spans.

Token accounting (count and log the EXACT list -- a silent off-by-one here
once cost us a training run):

  * :data:`TASK_TOKEN` = ``<SPORTS_UNIFIED>``                       -> 1 token
  * :data:`SCHEMA_TOKENS` (structural + flag)                       -> 13 tokens
  * ------------------------------------------------------------------------
  * :data:`SPORTS_CUSTOM_TOKENS` (everything registered)            -> 14 tokens

  The task marker is intentionally left UNANCHORED by semantic anchor init
  (it has no natural-language gloss), so expect ``seeded 13/14`` in the
  anchor-init log.
"""


from typing import Dict, List, Optional



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
# and NOT persisted by ``save_pretrained``) -- register the same expansion in
# both your training and evaluation entry points, or the encoder input
# silently diverges between them.
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
    :data:`TEAM_OTHER` (real GT carries both casings). Dirty / missing /
    undeterminable values (``""``, ``"Unknown"``, ``None``, anything
    unrecognised) collapse to :data:`UNKNOWN`. This is the fixed-slot
    categorical rule: the slot always emits exactly one of the four flag
    tokens, never free text.
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

# ======================================================================
# Finite-state constrained decoding
# ======================================================================
"""FSM-constrained decoding for the sports-unified schema.

Mirrors ``wider_constrained_decoding.py`` but for a schema that DOES have
free-text spans (sport / scene values, OCR text, and the ``<gdesc>`` body).
Free-text spans return "vocab minus structural tokens minus BOS/EOS/loc",
so the model can copy arbitrary text but can never emit a structural token or
a location bin mid-span; the terminator that closes each span is added back to
the allow-set by the grammar.

Target grammar (after Florence-2's forced ``</s><s>`` prefix)::

    <sport> {text}* <stype> {text}* (
        <player> <bbox> <loc>x4 </bbox> <team> {a|b|other|unknown}
            ( <ocr> {text}* <loc>x8 )*          # ocr_quads="after_text" (E1)
            ( <ocr> {text}* )*                  # ocr_quads="none"       (E1b)
        </player>
    )* <gdesc> {text}* </s>

``max_players`` caps the number of ``<player>`` blocks (the per-frame cap lives
in the grammar, not in indexed tags). ``<unknown>`` stays allowed for the team
flag at eval time (team can be genuinely undeterminable). ``ocr_quads`` selects
the OCR item shape and MUST match what the checkpoint was trained with (E1 vs
E1b); it is passed explicitly (weights carry no hint) and logged at build time.
"""


from typing import Callable, List

from loguru import logger



# --- FSM states -----------------------------------------------------------
_S_SPORT_OPEN = 0
_S_SPORT_VALUE = 1
_S_STYPE_OPEN = 2  # (unused fast-path; STYPE opener handled inside SPORT_VALUE)
_S_STYPE_VALUE = 3
_S_BBOX_OPEN = 4
_S_BBOX_LOC = 5
_S_BBOX_CLOSE = 6
_S_TEAM_OPEN = 7
_S_TEAM_VALUE = 8
_S_OCR_OR_CLOSE = 9
_S_OCR_TEXT = 10
_S_OCR_LOC = 11
_S_PLAYER_OR_GDESC = 12
_S_GDESC_VALUE = 13
_S_DONE = 14


def build_sports_prefix_allowed_tokens_fn(
    tokenizer,
    *,
    max_players: int = 9,
    allow_unknown_team: bool = True,
    ocr_quads: str = "after_text",
) -> Callable[[int, "object"], List[int]]:
    """Build a ``prefix_allowed_tokens_fn`` enforcing the sports grammar.

    Args:
        tokenizer: sports-extended Florence-2 tokenizer (all custom tokens +
            ``<loc_*>`` present).
        max_players: hard cap on the number of ``<player>`` blocks.
        allow_unknown_team: keep ``<unknown>`` in the team-flag alphabet
            (True for training-schema parity + honest eval).
        ocr_quads: OCR quad-supervision mode -- MUST match the serializer /
            ``schema_caps.json`` value the checkpoint was trained with (there is
            no way to introspect it from the weights, so it is passed explicitly
            and logged loudly). ``"after_text"`` (E1) emits ``<ocr>TEXT<loc*8>``;
            ``"none"`` (E1b) emits ``<ocr>TEXT`` with NO loc tokens -- the OCR-text
            span is then terminated by the next ``<ocr>`` or ``</player>``.
    """
    if ocr_quads not in ("after_text", "none"):
        raise ValueError(
            f"grammar ocr_quads must be 'after_text' or 'none'; got {ocr_quads!r} "
            f"('before_text'/E4 has no grammar branch yet)."
        )
    no_quads = ocr_quads == "none"

    def _tid(token: str) -> int:
        idx = tokenizer.convert_tokens_to_ids(token)
        unk = getattr(tokenizer, "unk_token_id", None)
        if idx is None or (unk is not None and idx == unk):
            raise ValueError(
                f"Token {token!r} missing from tokenizer vocab -- checkpoint "
                f"processor is missing the sports custom tokens."
            )
        return int(idx)

    sport_open = _tid(SPORT_OPEN)
    stype_open = _tid(STYPE_OPEN)
    gdesc_open = _tid(GDESC_OPEN)
    player_open = _tid(PLAYER_OPEN)
    player_close = _tid(PLAYER_CLOSE)
    bbox_open = _tid(BBOX_OPEN)
    bbox_close = _tid(BBOX_CLOSE)
    team_open = _tid(TEAM_OPEN)
    ocr_open = _tid(OCR_OPEN)

    team_ids = [_tid(TEAM_A), _tid(TEAM_B), _tid(TEAM_OTHER)]
    if allow_unknown_team:
        team_ids.append(_tid(UNKNOWN))

    loc_ids = [_tid(f"<loc_{i}>") for i in range(LOC_BINS)]
    loc_set = set(loc_ids)

    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("tokenizer.eos_token_id is None; cannot build grammar.")

    # ---- free-text alphabet = all vocab minus structural/flag/loc/specials -
    structural_ids = {_tid(t) for t in SPORTS_CUSTOM_TOKENS}
    exclude = set(structural_ids) | set(loc_set)
    for sid in (bos_id, eos_id, getattr(tokenizer, "pad_token_id", None),
                getattr(tokenizer, "unk_token_id", None)):
        if sid is not None:
            exclude.add(int(sid))
    vocab_size = len(tokenizer)
    free_text_ids = [i for i in range(vocab_size) if i not in exclude]

    # Precompute the free-text-plus-terminator allow-lists (returned by ref).
    allow_sport_value = free_text_ids + [stype_open]
    allow_stype_value = free_text_ids + [player_open, gdesc_open]
    # after_text: the OCR-text span is terminated by the first <loc>. none: it is
    # terminated by the next <ocr> (new item) or </player> (end of the block).
    allow_ocr_text = (
        free_text_ids + [ocr_open, player_close] if no_quads
        else free_text_ids + loc_ids
    )
    allow_gdesc_value = free_text_ids + [eos_id]
    allow_player_or_gdesc = [player_open, gdesc_open]
    allow_gdesc_only = [gdesc_open]
    allow_ocr_or_close = [ocr_open, player_close]

    logger.info(
        f"sports grammar built: {len(free_text_ids)} free-text ids, "
        f"max_players={max_players}, team_alphabet={len(team_ids)}, "
        f"ocr_quads={ocr_quads}"
    )

    def _replay(ids: List[int]):
        """Recover (state, loc_count, ocr_loc_count, n_players) from ids."""
        state = _S_SPORT_OPEN
        loc_count = 0
        ocr_loc = 0
        n_players = 0
        for t in ids:
            if state == _S_SPORT_OPEN:
                if t == sport_open:
                    state = _S_SPORT_VALUE
            elif state == _S_SPORT_VALUE:
                if t == stype_open:
                    state = _S_STYPE_VALUE
                # else: free text -> stay
            elif state == _S_STYPE_VALUE:
                if t == player_open:
                    state, loc_count = _S_BBOX_OPEN, 0
                elif t == gdesc_open:
                    state = _S_GDESC_VALUE
            elif state == _S_BBOX_OPEN:
                if t == bbox_open:
                    state, loc_count = _S_BBOX_LOC, 0
            elif state == _S_BBOX_LOC:
                if t in loc_set:
                    loc_count += 1
                    if loc_count >= 4:
                        state = _S_BBOX_CLOSE
            elif state == _S_BBOX_CLOSE:
                if t == bbox_close:
                    state = _S_TEAM_OPEN
            elif state == _S_TEAM_OPEN:
                if t == team_open:
                    state = _S_TEAM_VALUE
            elif state == _S_TEAM_VALUE:
                if t in team_ids:
                    state = _S_OCR_OR_CLOSE
            elif state == _S_OCR_OR_CLOSE:
                if t == ocr_open:
                    state, ocr_loc = _S_OCR_TEXT, 0
                elif t == player_close:
                    n_players += 1
                    state = _S_PLAYER_OR_GDESC
            elif state == _S_OCR_TEXT:
                if no_quads:
                    # none (E1b): no loc tokens. A new <ocr> starts the next item
                    # (stay in OCR_TEXT); </player> closes the block.
                    if t == ocr_open:
                        ocr_loc = 0
                    elif t == player_close:
                        n_players += 1
                        state = _S_PLAYER_OR_GDESC
                    # else free text -> stay
                elif t in loc_set:
                    ocr_loc = 1
                    state = _S_OCR_LOC
                    if ocr_loc >= 8:
                        state = _S_OCR_OR_CLOSE
                # else free text -> stay
            elif state == _S_OCR_LOC:
                if t in loc_set:
                    ocr_loc += 1
                    if ocr_loc >= 8:
                        state = _S_OCR_OR_CLOSE
            elif state == _S_PLAYER_OR_GDESC:
                if t == player_open:
                    state, loc_count = _S_BBOX_OPEN, 0
                elif t == gdesc_open:
                    state = _S_GDESC_VALUE
            elif state == _S_GDESC_VALUE:
                if t == eos_id:
                    state = _S_DONE
            elif state == _S_DONE:
                pass
        return state, loc_count, ocr_loc, n_players

    def prefix_allowed_tokens_fn(batch_id: int, input_ids) -> List[int]:  # noqa: ARG001
        ids = input_ids.tolist() if hasattr(input_ids, "tolist") else list(input_ids)

        # Respect Florence-2's forced BOS: emit BOS until it appears.
        if bos_id is not None:
            if bos_id not in ids:
                return [bos_id]
            start = len(ids) - 1 - ids[::-1].index(bos_id) + 1
        else:
            start = 0

        state, loc_count, ocr_loc, n_players = _replay(ids[start:])

        if state == _S_SPORT_OPEN:
            return [sport_open]
        if state == _S_SPORT_VALUE:
            return allow_sport_value
        if state == _S_STYPE_VALUE:
            return allow_stype_value
        if state == _S_BBOX_OPEN:
            return [bbox_open]
        if state == _S_BBOX_LOC:
            return loc_ids
        if state == _S_BBOX_CLOSE:
            return [bbox_close]
        if state == _S_TEAM_OPEN:
            return [team_open]
        if state == _S_TEAM_VALUE:
            return team_ids
        if state == _S_OCR_OR_CLOSE:
            return allow_ocr_or_close
        if state == _S_OCR_TEXT:
            return allow_ocr_text
        if state == _S_OCR_LOC:
            return loc_ids
        if state == _S_PLAYER_OR_GDESC:
            return allow_gdesc_only if n_players >= max_players else allow_player_or_gdesc
        if state == _S_GDESC_VALUE:
            return allow_gdesc_value
        # _S_DONE
        return [eos_id]

    return prefix_allowed_tokens_fn


__all__ = ["build_sports_prefix_allowed_tokens_fn"]
