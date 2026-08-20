# Native Association

> Companion release for **"Native Association: Confidence-Aware Human Perception in the Wild with a Foundation VLM"** — Igal Dmitriev & Ofir Liba (WSC Sports), 1st Workshop on Human-Centered Multimodal Intelligence in the Wild (**HCMIW @ ECCV 2026**), Malmö, Sept 8. [Paper + reviews on OpenReview](https://openreview.net/forum?id=ZYKiyXfSy8).

Extracting *who is where, on which team, wearing which number* from a broadcast frame is normally done by stitching a detector, an OCR engine and classifiers together — and under occlusion the stitch attaches a correctly read number to the **wrong** athlete. This repo releases the mechanism that removes that step: a **finite-state grammar** under which a 0.77B VLM (Florence-2) emits *all* per-person attributes inside each owner's block of **one** decoded sequence, so association is a property of decoding, and a **one-replay per-field confidence** that turns the same pass into a reject option.

| If you want to... | Go to |
|---|---|
| **Read the paper, supplementary, or poster** | [`paper/`](paper/) — camera-ready PDFs + the HCMIW poster |
| **Use the grammar / parser / serializer / metrics in code** | [`src/native_association/`](src/native_association/) |
| **Re-run the API baselines with the exact prompt** | [`prompts/gemini_prompt_eccv_v1.md`](prompts/gemini_prompt_eccv_v1.md) |
| **Map paper claims to released components** | [`docs/REPRODUCING.md`](docs/REPRODUCING.md) |

## Headline results (frozen 567-frame multi-sport test)

| Metric | Ours (E1b) | Gemini-2.5-pro | Gemini-3.1-pro |
|---|---|---|---|
| Detection F1 | **0.950** | 0.652 | 0.745 |
| Jersey F1 (any-OCR) | **0.757** | 0.704 | 0.737 |
| Jersey-digit AER ↓ | **0.057** | 0.242 | 0.214 |
| Schema-valid (%) | **100** | – | – |
| Latency (ms/frame) ↓ | **1224** | 26009 | 8426 |

One extra teacher-forced pass yields a per-field confidence (AUROC 0.879) supporting a reject option: jersey precision **0.71 → 0.96** at half coverage. The identical recipe transplanted to public **WIDER-Attribute** reaches **93.1 mAP** given-box and **84.5 mAP** end-to-end (to our knowledge the first detection-coupled result under its standard test protocol).

## What's in `src/native_association/`

Self-contained (imports rewritten from the internal codebase; no proprietary dependencies):

* [`constrained_decoding.py`](src/native_association/constrained_decoding.py) — **the finite-state grammar**: a `prefix_allowed_tokens_fn` for `generate()` that admits only grammar-legal continuations, making every decode schema-valid *by construction* and carrying association (each attribute is emitted inside exactly one `<player>` block).
* [`sports_tokens.py`](src/native_association/sports_tokens.py) — the atomic structural tokens (`<sport>`, `<player>`, `<bbox>`, `<team>` + flags, `<ocr>`, `<gdesc>`) and their registration.
* [`serializer.py`](src/native_association/serializer.py) — annotation → target-string serialization. **The schema is locked in this module**: players largest-first, OCR items largest-quad-first, caps at 12 players / 17 OCR items.
* [`output_parser.py`](src/native_association/output_parser.py) — decoded sequence → structured records, plus schema-compliance validation.
* [`confidence_replay.py`](src/native_association/confidence_replay.py) — the one-pass **unmasked** teacher-forced replay (paper Eq. 1) and the geo/arith/min aggregators from the supplementary's ablation.
* [`metrics.py`](src/native_association/metrics.py) — the evaluation suite: Hungarian IoU matching, any-OCR / strict jersey rules, permutation-invariant team purity, fuzzy OCR-text F1, and the **association error rate (AER)** with its decomposition.
* [`sport_vocab.py`](src/native_association/sport_vocab.py), [`ocr_text.py`](src/native_association/ocr_text.py), [`constants.py`](src/native_association/constants.py), [`coordinate_utils.py`](src/native_association/coordinate_utils.py) — label canonicalization, OCR-text normalization ('?' handling), and the 1000-bin coordinate quantization.

Dependencies for the code: `torch`, `numpy`, `scipy`, `python-Levenshtein`, `loguru` (plus `transformers` for the model itself).

## Release scope

Per the paper (Sec. 4, *Reproducibility*): the sports corpus and sports checkpoint are **proprietary and not released**. This repo carries the evaluation suite, the grammar and serialization, the confidence recipe, and the verbatim API prompts. The WIDER-Attribute results are reproducible end-to-end from public data with the recipe here; the per-system prediction caches and the WIDER checkpoint are being prepared for release — see [`docs/REPRODUCING.md`](docs/REPRODUCING.md) for the current status of each artifact.

## Related

* [`florence2-unified-perception`](https://github.com/Igal20/florence2-unified-perception) — the companion methodology bundle: how to expand Florence-2's vocabulary (Medium article) and the two-stage fine-tuning recipe (IMVC 2026 talk). This repo is the *research-grade* continuation: grammar-guaranteed validity, native association, and calibrated per-field confidence.

## Citation

```bibtex
@inproceedings{dmitriev2026native,
  title     = {Native Association: Confidence-Aware Human Perception in the Wild with a Foundation VLM},
  author    = {Dmitriev, Igal and Liba, Ofir},
  booktitle = {ECCV Workshops (Human-Centered Multimodal Intelligence in the Wild)},
  year      = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
