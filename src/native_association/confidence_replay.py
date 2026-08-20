"""Per-field confidence from ONE unmasked teacher-forced replay (paper Sec. 3.2).

After constrained generation, a single teacher-forced forward pass over the
generated tokens recovers per-token log-probabilities — no second generation,
no auxiliary head (~150 ms, ~10% of generation). The replay is UNMASKED:
confidences are full-vocabulary probabilities (the model's raw beliefs), not
values renormalized over the few grammar-legal tokens.

For any contiguous field F = (y_t1, ..., y_tk) the confidence is the
geometric mean of its per-token probabilities:

    c(F) = exp( (1/k) * sum_i log P(y_ti | y_<ti, X) )        (Eq. 1)

The supplementary's aggregation ablation shows geometric mean, arithmetic
mean and min-probability coincide to three decimals on jersey fields (1–2 BPE
tokens): on single-token fields every aggregator is identical by definition.
Eq. 1 is kept as the general length-normalized form.

This module is a self-contained reference implementation extracted from the
paper's evaluation code. Feed it the ``generate()`` output ids and a token-id
mapping; it returns, per emitted player block, the per-token probabilities of
each OCR item's text tokens plus the standard aggregations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch

AGGREGATORS = ("geo", "arith", "min")


def aggregate(probs: List[float]) -> Dict[str, Optional[float]]:
    """Standard aggregations of a field's per-token probabilities."""
    if not probs:
        return {a: None for a in AGGREGATORS}
    p = np.asarray(probs, dtype=float)
    logp = np.log(np.clip(p, 1e-12, 1.0))
    return {
        "geo": float(np.exp(logp.mean())),   # Eq. 1 (paper default)
        "arith": float(p.mean()),
        "min": float(p.min()),
    }


@torch.no_grad()
def replay_text_token_probs(
    model,
    inputs: Dict[str, torch.Tensor],
    gen_ids: torch.Tensor,
    tk: Dict[str, Any],
) -> List[List[List[float]]]:
    """One unmasked teacher-forced replay -> per-player OCR-text token probs.

    Args:
        model:   the fine-tuned Florence-2-style encoder-decoder.
        inputs:  the processor output used for generation
                 (``input_ids``, ``pixel_values``).
        gen_ids: the constrained ``generate()`` output ids, shape [1, T].
        tk:      token-id mapping with keys ``loc_set`` (set of all
                 ``<loc_*>`` ids), ``player_open``, ``player_close``,
                 ``ocr_open``.

    Returns:
        Per player block (emission order): a list of OCR items, each a list
        of that item's per-text-token full-vocabulary probabilities.
    """
    out = model(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        decoder_input_ids=gen_ids,
    )
    logp = torch.log_softmax(out.logits.float(), dim=-1)[0]  # [T, V]

    seq = gen_ids[0].tolist()
    n = len(seq)
    loc_set = tk["loc_set"]
    player_open, player_close = tk["player_open"], tk["player_close"]
    ocr_open = tk["ocr_open"]

    players: List[List[List[float]]] = []
    i = 0
    while i < n:
        if seq[i] != player_open:
            i += 1
            continue
        ocr_probs: List[List[float]] = []
        j = i + 1
        while j < n and seq[j] != player_open:
            tj = seq[j]
            if tj == ocr_open:
                m = j + 1
                text_pos: List[int] = []
                while (
                    m < n
                    and seq[m] not in loc_set
                    and seq[m] not in (ocr_open, player_close, player_open)
                ):
                    text_pos.append(m)
                    m += 1
                # skip optional per-OCR quad loc tokens (E1 schema only)
                k = 0
                while m < n and k < 8 and seq[m] in loc_set:
                    m += 1
                    k += 1
                probs = [
                    float(np.exp(logp[p - 1, seq[p]].item()))
                    for p in text_pos
                    if p - 1 >= 0
                ]
                ocr_probs.append(probs)
            if tj == player_close:
                j += 1
                break
            j += 1
        players.append(ocr_probs)
        i = j
    return players
