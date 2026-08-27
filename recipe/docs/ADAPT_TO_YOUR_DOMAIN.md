# Adapting the recipe to your domain

Five steps, one worked example (WIDER-Attribute — whose checkpoint this repo releases), and the honesty checklist that should travel with any port.

## The five-step port

1. **Redefine the slot vocabulary.** Keep the skeleton (`<entity>…</entity>` blocks inside one sequence); swap the per-entity attributes. Sports: team flag + jersey/OCR text. Pedestrian attributes: 14 binary flags. Retail: price OCR + stock state. Each categorical attribute = a small closed set of single tokens.
2. **Redesign the grammar table** (see SCHEMA_AND_GRAMMAR.md): states, budgets, caps. Set entity/item caps from your data's observed maxima — then verify they never bind.
3. **Rewrite the serializer** for your annotation format — validity gates, canonical largest-first order, your `?`-equivalent policy. This is the ONE bespoke artifact; `reference/serializer.py` is the worked pattern.
4. **Recheck the decode defaults** (`no_repeat_ngram_size=0`, `early_stopping=False`) — repeated schema fragments make n-gram bans fatal in any domain.
5. **Re-derive the evaluation** — your AER analogue is "correctly-read attribute bound to the wrong entity"; keep the decomposition and the frozen-test discipline.

What does NOT change: frozen encoders, uniform CE, warm-started token rows, grammar-constrained validation decoding, the unmasked replay, Eq. 1.

## Worked example: WIDER-Attribute

The transplant that tests whether this is a recipe rather than a lucky checkpoint. The ONLY change was step 1: team/jersey/scene slots became 14 binary attribute flags per person. Fine-tuned only on the WIDER training split (not zero-shot from sports weights) — a re-instantiation of the same procedure.

Results (test: 6,918 images, 29,177 person instances):

| Setting | mAP |
|---|---|
| A decade of task-specific specialists (given-box) | 80.5 – 87.5 |
| **This recipe, given-box** (whole-image + per-crop late fusion) | **93.1** |
| **This recipe, end-to-end** (no GT boxes at any stage) | **84.5** — to our knowledge the first detection-coupled result under the standard protocol |

The phenomenology replicated too: tuning null (+0.02 mAP), usable confidence (error 4.1% → 1.5% at 10% abstention; ECE 0.028 after fusion), structural signals route-not-filter. Scope it honestly: WIDER's protocol tests the *recipe's generality*, not the text–entity association claim itself.


![WIDER-Attribute qualitative grid: the same recipe, 14 binary attributes per person](../../assets/wider_qual_grid.png)

### The WIDER checkpoint

**The fine-tuned WIDER-Attribute checkpoint is released on Hugging Face: [Igal20/native-association-wider-florence2](https://huggingface.co/Igal20/native-association-wider-florence2)** (weights + tokenizer + a model card with the exact loading snippet, including the `lm_head.pth` overlay). With it, the given-box and end-to-end numbers above are reproducible from public data end-to-end: WIDER images + this checkpoint + `reference/grammar.py` decoding + `reference/metrics.py` scoring.

## The honesty checklist (ships with every port)

- [ ] Entity cap audited as non-binding on your data
- [ ] Source-level split leakage measured and stated (not just frame-level dedup)
- [ ] Single-seed? Pre-register a noise floor and scope tuning claims to "no measurable gain under the tested configurations"
- [ ] Both halves of the re-query trade reported (recall gained AND association cost)
- [ ] Headline-metric exclusions (your team-bleed analogue) stated in the main text, with the alternative ranking
- [ ] CIs that span zero printed as prominently as those that don't
