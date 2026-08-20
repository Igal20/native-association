# Schema & grammar — the annotation contract and the state machine

The single most important design act in this recipe: decide the **exact token string** the decoder emits for one image, then make that string the *only* thing the decoder can emit.

## 1. The serialization (sports instantiation)

One image → one sequence:

```
<sport>{s} <stype>{t}
  ( <player> <bbox> <loc>×4 </bbox> <team>{flag}
        ( <ocr>{text} [<loc>×8] )*
    </player> )*
<gdesc>{desc} </s>
```

Design rules that carry the guarantees — port these, not the sports specifics:

- **Typed, single-token keys.** Each structural key (`<sport>`, `<player>`, `<bbox>`, `<team>`, `<ocr>`, `<gdesc>`) is ONE added token. Literal JSON costs 3–5 BPE tokens per key and is malformable; a typed serialization is neither.
- **Attributes live inside their owner's block.** Everything between `<player>` and `</player>` belongs to that player. That containment IS the association mechanism — no post-hoc binding step exists to get wrong.
- **Deterministic canonical order.** Entities largest-first (bbox area, descending); per-entity list items (OCR reads) largest-quad-first. Under the grammar we observed none of the repetition/early-termination failures that motivated Pix2Seq's randomized ordering.
- **Coordinates quantized** to 1000 bins, emitted as existing `<loc_0>…<loc_999>` tokens (Florence-2 already has them — reuse, don't add).
- **Bounded repetition.** The grammar caps entities (12 players) and per-entity items (17 OCR reads). Set the caps from your data's observed maximum so they never bind (audit this!).
- **Categorical values as closed token sets.** Team is one of four single-token flags. Free text (OCR strings, the scene description) stays ordinary BPE text.

The `?`-policy: our annotators mark unreadable characters with `?`. Choose one of `clean` (strip `?`, keep the residue if non-empty — paper default), `drop` (drop any `?`-containing item), `keep` (train the model to reproduce uncertainty verbatim). Apply the SAME policy to ground truth at evaluation time. See `reference/serializer.py`.

## 2. The finite-state grammar

Constrained decoding via a `prefix_allowed_tokens_fn` passed to `generate()`: at each step, compute the set of legal next tokens from the current state; every illegal token gets probability zero.

States and transitions (see `reference/grammar.py` for the executable version):

| State | Legal continuations |
|---|---|
| start | `<sport>` only — the first-token mask; this alone prevents the start-token collapse below |
| after `<sport>` | sport-value tokens, then `<stype>` |
| after `<stype>` value | `<player>` or `<gdesc>` |
| in `<player>`, pre-bbox | `<bbox>` |
| in `<bbox>` | exactly 4 `<loc_*>` tokens, then `</bbox>` |
| after `</bbox>` | `<team>` |
| after `<team>` | exactly one of the 4 team-flag tokens |
| after team flag | `<ocr>` or `</player>` |
| in `<ocr>` text | free-text tokens; exit on `<ocr>` / `</player>` (and optionally 8 `<loc_*>` for quads) |
| after `</player>` | `<player>` (if under the entity cap) or `<gdesc>` |
| in `<gdesc>` | free text until `</s>` |

Two properties follow **by construction**: every decode parses (validity 1.0), and every attribute has exactly one owner.

**Why the grammar is load-bearing, not decoration:** decoding the *same* checkpoint with the constraint disabled collapses into an unbounded start-token loop — zero content tokens, zero parseable entities, 7× the latency — on both our sports data and WIDER. Beam search and n-gram bans change *how* it fails, not *whether* (they produce malformed output instead). Scope this correctly: that ablation isolates the grammar's contribution to decoding stability and format validity; it says nothing about perception accuracy.

**Practical trap (Florence-2 specific):** the language model inherits `no_repeat_ngram_size=3` and `early_stopping=True` from its text config. A trigram ban makes the *second* occurrence of any repeated schema fragment (`</player><player><bbox>…`) illegal, silently capping structured output at 1–2 entities. Set `no_repeat_ngram_size=0`, `early_stopping=False` in every generate call and persist a cleaned `generation_config` with the checkpoint.

## 3. Token registration

Add the structural tokens as **atomic** tokens; warm-start each new embedding row with the mean embedding of descriptive sub-words (e.g. `<player>` ← "player"). Token-addition mechanics, tied-embedding pitfalls included, are worked through in the companion repo: [florence2-unified-perception](https://github.com/Igal20/florence2-unified-perception).
