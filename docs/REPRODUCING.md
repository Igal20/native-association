# Reproducing the paper — claim → artifact map

Status of every release the paper commits to (Sec. 4, *Reproducibility*),
and where each paper claim is carried in this repo.

| Paper artifact | Status | Where |
|---|---|---|
| Finite-state grammar + inference harness | **Released** | `reference/grammar.py` (+ spec in `recipe/docs/SCHEMA_AND_GRAMMAR.md`) |
| Schema / serialization (locked) | **Released** | `reference/serializer.py` |
| Evaluation suite (matching, jersey rules, AER + decomposition) | **Released** | `reference/metrics.py`, `reference/output_parser.py` |
| Per-field confidence recipe (Eq. 1, unmasked replay) | **Released** | `reference/confidence_replay.py` (+ `recipe/docs/CONFIDENCE_AND_REQUERY.md`) |
| Training recipe | **Released (prose)** | `recipe/docs/TRAINING.md` — written to be scaffolded into a runnable script by a coding assistant |
| Verbatim API prompts + retry policy | **Released** | `prompts/gemini_prompt_eccv_v1.md` |
| **WIDER-Attribute checkpoint** | **To be released on this repo's Releases page** | with it: WIDER given-box (93.1 mAP) and end-to-end (84.5 mAP) reproduce from public data |
| Per-system sports prediction caches (re-scorable) | Being prepared | — |
| Sports corpus + sports checkpoint | **Not released** (proprietary) | see paper Sec. 4 |

## Claim → component

| Claim (camera-ready) | Carried by |
|---|---|
| Schema validity 1.0 by construction (Sec. 3.1) | `reference/grammar.py` — only grammar-legal continuations are reachable |
| Native association: every read inside exactly one `<player>` block | grammar states + serialization order (`reference/serializer.py`) |
| AER definition + decomposition (Sec. 4; supp. Sec. 3.1) | `reference/metrics.py` (digit-swap / digit-unmatched / team-bleed) |
| Any-OCR vs strict jersey rule (supp. Tab. S12) | `reference/metrics.py` |
| Team purity, permutation-invariant (Sec. 4; supp. Sec. 7) | `reference/metrics.py` |
| Eq. 1 confidence; unmasked replay (Sec. 3.2) | `reference/confidence_replay.py` |
| Aggregator ablation: geo = arith = min to 3 dp (supp. Tab. S7) | `confidence_replay.aggregate` |
| Grammar collapse without constraint (Sec. 5.3; supp. Tab. S16–S17) | reproduce by decoding any checkpoint with the prefix function disabled |
| 1000-bin quantization, decode-default traps (supp. Sec. 9) | `recipe/docs/SCHEMA_AND_GRAMMAR.md` |
| API baseline protocol: temperature 0, ≤3 retries at +0.3 (supp. Sec. 5) | `prompts/gemini_prompt_eccv_v1.md` |

## Frozen-test discipline

All sports numbers in the paper are computed once on a frozen 567-frame split
under a fingerprint guard (frame count + frame-set hash + annotation hash,
abort on mismatch); every analysis is an offline re-score of those cached
predictions. The discipline itself is documented in
`recipe/docs/EVALUATION.md` so ports inherit it.
