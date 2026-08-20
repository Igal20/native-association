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

from __future__ import annotations

from typing import Callable, List

from loguru import logger

from .constants import LOC_BINS

from .sports_tokens import (
    BBOX_CLOSE,
    BBOX_OPEN,
    GDESC_OPEN,
    OCR_OPEN,
    PLAYER_CLOSE,
    PLAYER_OPEN,
    SPORT_OPEN,
    SPORTS_CUSTOM_TOKENS,
    STYPE_OPEN,
    TEAM_A,
    TEAM_B,
    TEAM_OPEN,
    TEAM_OTHER,
    UNKNOWN,
)

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
