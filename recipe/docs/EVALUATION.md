# Evaluation — one suite, every system, and the metric that matters

Score every system — yours, API baselines, cascades — through **one suite with identical matching**, or the comparison is fiction. Reference implementation: `reference/metrics.py`.

## The standard metrics

- **Detection:** Hungarian assignment at IoU ≥ 0.5 → P/R/F1. *Match coverage* (fraction of GT entities matched) equals detection recall and is the honest denominator for everything downstream.
- **Categorical attributes (e.g. team):** report BOTH exact accuracy (scores the annotator's A/B convention) and a permutation-invariant purity (per-frame Hungarian between predicted and GT labels) — cluster labels are arbitrary, so a systematic flip is not a semantic error. Print the denominators; three team numbers with unstated denominators cost us a review round.
- **Entity text (jersey):** an *any-item* rule (GT number correct if it appears as any emitted digit token for the matched entity) with a *strict exact-string* variant alongside. Check ranking invariance across both rules.
- **Free text:** fuzzy Hungarian at normalized Levenshtein τ = 0.7, pooled per frame, deliberately association-blind — binding errors are charged to AER, not text F1.
- **Output validity:** parse-success / empty-output / schema-valid rates. For API baselines also count **first-attempt** parse failures (post-retry numbers hide fragility) and degenerate geometry (zero-area boxes — Gemini-3.1 emitted one on 20.3% of predicted players).

## AER — the association error rate

The metric this recipe exists for: **the fraction of correctly-read fields not bound to their true owner**, covering both binding-to-a-wrong-matched-entity and grounding-to-no-valid-box. Decompose it:

```
all-text misassociation = digit-swap + digit-unmatched + team-bleed
```

- *swap* = read landed on a matched-but-wrong entity (a genuine identity error);
- *unmatched* = read grounded to nothing (the API failure class: 90–97% of their misbindings);
- *bleed* = text identical across entities (team names) — owner ill-defined; **exclude it from the headline** and say so in the main text, because the exclusion changes rankings (our cascades beat us under all-text AER; the digit-only ranking is specific to that choice).

Also report *matched-only* AER for transparency but never as the headline: it rewards a system for grounding a read to nothing rather than to the wrong entity.

## Discipline that makes the numbers defensible

- **Frozen test + fingerprint guard** (frame count, frame-set hash, annotation hash; abort on mismatch). Decode once per variant; all analyses re-score cached predictions.
- **Paired bootstrap** (10k resamples of frames, the same resample scoring every system) for every headline delta — and print the intervals that span zero as prominently as the ones that don't. Ours: jersey-F1 vs the best API +0.02 [−0.02, +0.06]; digit-AER vs oracle-boxed cascades indistinguishable. Papers that volunteer this survive review; papers that don't get it found.
- **Size stratification:** bucket matched entities into GT-box-area quartiles; report detection recall and text F1 per bucket. This is where the real error budget lives and where the re-query earns its keep.
- **A reconciliation gate before any analysis:** a per-record recomputation must reproduce the published headline numbers exactly, or the analysis run aborts. Cheap, and it catches silent definition drift.
