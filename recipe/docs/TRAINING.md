# Training — deliberately plain

The recipe's counter-intuitive core finding: **the gains live in the structure, not in added weights.** Train the simplest thing that can learn the grammar, then stop.

## The recipe

| Knob | Setting | Why |
|---|---|---|
| Backbone | Florence-2-large (0.77B), DaViT vision tower + BART encoder–decoder | any similar VLM with a token decoder should port |
| Frozen | vision tower AND text encoder | pretrained visual priors preserved; training teaches the decoder the *output schema* — "vocabulary alignment", not representation learning |
| Trained | BART decoder (full fine-tune) + shared token embeddings (incl. the warm-started structural rows) | the embedding matrix is tied across encoder/decoder/output projection — one table |
| Loss | uniform cross-entropy | no token weighting, no hierarchical loss, no curriculum |
| Optimizer | AdamW, lr 5e-5, weight decay 0.1, cosine schedule, 5% warm-up | |
| Batch / epochs | 6 / up to 20, early stop on val loss (patience 3) | ~4.6k train images |
| Grammar in training | **none** — the teacher-forced loss never sees the grammar | the grammar is an inference-time guarantee; validation *decoding* for checkpoint selection IS grammar-constrained from epoch 1 |
| Selection | aggregate `val_score` = unweighted mean of your task metrics (ours: detection F1, team acc, jersey F1, OCR-text F1, scene acc) | pick `best_model_score` |
| Precision | fp32 for training and accuracy metrics; fp16 for reported latency | |
| Seed | one, with a **pre-registered noise floor** (+0.005 on the 0–1 aggregate): deltas under the floor are noise, and you commit to that BEFORE looking | this is what lets a small study make honest claims |

## What NOT to spend time on (measured, twice)

**Adapters.** From the trained decoder we added LoRA of increasing capacity (decoder V / vision / decoder QVO / decoder QKVO, up to 3.15M params) and re-selected on validation: best delta **+0.0011**, ~5× under the pre-registered floor. The identical null replicated on WIDER-Attribute (+0.02 mAP). Scope honestly: *no measurable gain under the tested adaptation configurations* — not "tuning never works". But budget accordingly.

**What DID matter, for contrast:** the single structural decision of dropping per-OCR quad supervision (emit text only) was worth **+0.008** aggregate — ~7× the best tuning delta — plus 28% latency. Structure beats capacity in this regime (frozen towers, sub-1B decoder, ~5k images).

## Data discipline (do this before training anything)

- **Freeze the test set** and decode it once per model variant; every analysis afterwards is an offline re-score of cached predictions.
- Guard it with a **fingerprint**: frame count + frame-set hash + annotation-content hash, recomputed by every evaluation, abort on mismatch. This is ~30 lines and it will save you from yourself.
- Split at the frame level keyed on a perceptual hash (identical/near-duplicate frames can't straddle splits), and **audit source-level leakage** from whatever provenance you have (we found 4/567 test frames sharing a source video with train — measured and disclosed, not discovered by a reviewer).
- Cap audit: confirm the entity cap never binds on your data (`max(entities per image) <= cap`).
