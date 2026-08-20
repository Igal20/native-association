# Reproducing the paper — claim → artifact map

Status of every release the paper commits to (Sec. 4, *Reproducibility*),
and where each paper claim is carried in this repo.

| Paper artifact | Status | Where |
|---|---|---|
| Finite-state grammar + inference harness | **Released** | `src/native_association/constrained_decoding.py` |
| Schema / serialization (locked) | **Released** | `src/native_association/serializer.py`, `sports_tokens.py` |
| Evaluation suite (matching, jersey rules, AER + decomposition) | **Released** | `src/native_association/metrics.py`, `output_parser.py` |
| Per-field confidence recipe (Eq. 1, unmasked replay) | **Released** | `src/native_association/confidence_replay.py` |
| Verbatim API prompts + retry policy | **Released** | `prompts/gemini_prompt_eccv_v1.md` |
| Per-system prediction caches (sports, re-scorable) | *Being prepared* | — |
| WIDER-Attribute checkpoint + training recipe | *Being prepared* | — |
| Sports corpus + sports checkpoint | **Not released** (proprietary) | see paper Sec. 4 |

## Claim → component

| Claim (camera-ready) | Carried by |
|---|---|
| Schema validity 1.0 by construction (Sec. 3.1) | `constrained_decoding.build_sports_prefix_allowed_tokens_fn` — only grammar-legal continuations are reachable |
| Native association: every read inside exactly one `<player>` block | grammar states in `constrained_decoding.py`; serialization order in `serializer.py` |
| AER definition + decomposition (Sec. 4; supp. Sec. 3.1) | `metrics.py` (digit-swap / digit-unmatched / team-bleed) |
| Any-OCR vs strict jersey rule (Sec. 4; supp. Tab. S12) | `metrics.py` |
| Team purity, permutation-invariant (Sec. 4; supp. Sec. 7) | `metrics.py` |
| Eq. 1 per-field confidence; unmasked replay (Sec. 3.2) | `confidence_replay.py` |
| Aggregator ablation: geo = arith = min to 3 dp (supp. Tab. S7) | `confidence_replay.aggregate` (all three) |
| 1000-bin coordinate quantization (Sec. 3.1) | `constants.LOC_BINS`, `coordinate_utils.py` |
| '?' annotator-uncertainty handling (supp.) | `ocr_text.py` |
| API baseline protocol: temperature 0, ≤3 retries at +0.3 (supp. Sec. 5) | `prompts/gemini_prompt_eccv_v1.md` |

## Frozen-test discipline

All sports numbers in the paper are computed once on a frozen 567-frame split
under a fingerprint guard (frame count + frame-set hash + annotation hash,
abort on mismatch). The released caches will include the fingerprint so any
re-score is provably against the published test set.
