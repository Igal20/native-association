# Per-field confidence & the crop re-query

## One unmasked replay = confidence for every field

After constrained generation, run ONE teacher-forced forward pass over the generated tokens (feed `generate()`'s output as `decoder_input_ids`). That recovers per-token log-probabilities for the whole sequence at once — no second generation, no auxiliary head; ~150 ms, ~10% of generation cost, constant in sequence length because it's a single parallel forward.

**Replay UNMASKED.** Generation-time scores are grammar-masked, i.e. renormalized over the few legal tokens — useless as beliefs. The replay must use the full-vocabulary softmax. (This also explains why "objectness" margins computed at a two-way-legal decision saturate near 1: renormalization over two options is not confidence.)

For a contiguous field F = (y_t1 … y_tk):

```
c(F) = exp( (1/k) · Σ_i log P(y_ti | y_<ti, X) )        # Eq. 1 — geometric mean
```

Reference implementation: `reference/confidence_replay.py`.

**Aggregator choice is a non-issue for short fields** — measured: geometric mean, arithmetic mean and min-probability coincide to three decimals over 952 jersey reads (fields of 1–2 BPE tokens are aggregator-invariant by definition). Keep Eq. 1 as the general form; it matters only for longer text fields.

## What the confidence is good for — and what it is NOT

- **Reject option.** Rank predictions by field confidence, abstain on the least confident: jersey precision 0.71 (full coverage) → 0.87 (top 70%) → **0.96** (top half); AUROC 0.879. This relies only on the *ordering*, not calibration.
- **Calibration caveat.** The raw likelihood is over-confident (ECE 0.12–0.15 on sports). One temperature fixes it if you need absolute probabilities. (On WIDER after fusion it was calibrated out of the box, ECE 0.028 — domain-dependent.)
- **Negative results, so you don't rediscover them:** the structural `loc_conf` (geomean over box loc tokens; AUROC ≈ 0.51) and the objectness margin are near-random *correctness* predictors on both domains. They are routers, not filters — useful only to decide WHERE to re-query.

## The training-free crop re-query

Small entities are the dominant error mode, and it is a *resolution* bound, not a knowledge bound (a distant jersey spans 3–4 of the 1000 coordinate bins).

Procedure — no training, same model:

1. From the first pass, select entities that are **small** (bottom quartile by predicted box area) or **uncertain** (bottom quartile by field confidence). Thresholds validation-chosen; no test label touches selection.
2. Crop a padded region around each (padding fraction validation-chosen; ours 0.15).
3. Re-run the SAME model on the crop under the SAME grammar **capped at one entity**.
4. **Replace attributes in place** — never re-localize boxes, so duplicates cannot arise. Tag each entity with its source (frame / crop).

Cost: crops decode as short single-entity sequences → ~1.9 passes/image average, +50 ms.

Measured trade — state BOTH halves when you report it: tiny-bucket jersey F1 **0.61 → 0.75** (63 reads gained, 26 lost), while digit-AER worsens **0.057 → 0.068** — a crop straddling a pile-up can bind a neighbour's number. The re-query buys recall on the broken bucket at a small association cost; larger buckets are untouched.
