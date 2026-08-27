---
license: mit
base_model: microsoft/Florence-2-large
pipeline_tag: image-text-to-text
tags:
- florence-2
- vision-language-model
- person-attribute-recognition
- pedestrian-attributes
- grammar-constrained-decoding
- wider-attribute
- eccv-2026
library_name: transformers
---

# Native Association — WIDER-Attribute checkpoint (Florence-2-large, fixed-slot grammar)

The fine-tuned checkpoint behind the WIDER-Attribute results of
**"Native Association: Confidence-Aware Human Perception in the Wild with a Foundation VLM"**
(Igal Dmitriev & Ofir Liba, WSC Sports — HCMIW workshop @ ECCV 2026).
Recipe, reference implementations, and paper: **[github.com/Igal20/native-association](https://github.com/Igal20/native-association)** · [OpenReview](https://openreview.net/forum?id=ZYKiyXfSy8)

Florence-2-large (0.77B) fine-tuned to emit **all person attributes inside each person's block of one grammar-constrained sequence**: 14 binary WIDER attributes as fixed-slot single tokens per person, boxes as `<loc>` bins. Association is a property of decoding — every output is schema-valid by construction. Encoders frozen; BART decoder + shared embeddings trained; uniform CE; seed 42; best-validation-mA selection (epoch 9).

## Results (WIDER-Attribute test: 6,918 images / 29,177 person instances)

Recorded evaluations of this exact checkpoint (`evaluation/wider_eval_test_*.json` at training time):

| Mode | mAP | mA |
|---|---|---|
| Whole-image, single pass (end-to-end reading of all persons) | 82.9 | 79.4 |
| End-to-end two-stage (no GT boxes at any stage) | 84.4 | 80.0 |
| Given-box, per-crop | 91.9 | 91.4 |
| Given-box, whole-image (scene) | 90.1 | 90.0 |
| **Given-box, whole-image + per-crop late fusion (paper headline)** | **93.1** | — |

Reference points: a decade of task-specific given-box specialists span 80.5–87.5 mAP.
The end-to-end result is, to our knowledge, the first detection-coupled result reported
under WIDER-Attribute's standard protocol. The late-fusion numbers are the offline
re-scoring of the per-mode predictions (paper Sec. 5; supplementary).

## Loading

The custom Florence-2 code is pulled from `microsoft/Florence-2-large` (`trust_remote_code`).
**Important:** the language-model head with the expanded vocabulary is stored separately in
`lm_head.pth` — overlay it after `from_pretrained`, or logits for the custom schema tokens
will be wrong:

```python
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoProcessor

model_dir = Path("...")  # this repo, downloaded
device = "cuda"

model = AutoModelForCausalLM.from_pretrained(
    str(model_dir), trust_remote_code=True,
    torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
).to(device).eval()
processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True)

sd = torch.load(str(model_dir / "lm_head.pth"), map_location="cpu")
model.language_model.lm_head.load_state_dict(sd)
```

Decode **grammar-constrained** (`prefix_allowed_tokens_fn`) with `no_repeat_ngram_size=0` and
`early_stopping=False` — see
[`reference/grammar.py`](https://github.com/Igal20/native-association/blob/main/reference/grammar.py)
and the schema/grammar notes in the
[recipe](https://github.com/Igal20/native-association/tree/main/recipe). Decoding without the
constraint collapses (unbounded start-token loop) — the grammar is load-bearing.

## Intended use & limitations

Research checkpoint for reproducing the paper's public-benchmark results and for studying
grammar-constrained structured perception. Trained only on the WIDER-Attribute training
split (5,509 images); attribute vocabulary is WIDER's 14 binary attributes; not intended
for production person-analysis or surveillance use. The sports-broadcast corpus and
checkpoint from the same paper are proprietary and not released.

## Citation

```bibtex
@inproceedings{dmitriev2026native,
  title     = {Native Association: Confidence-Aware Human Perception in the Wild with a Foundation VLM},
  author    = {Dmitriev, Igal and Liba, Ofir},
  booktitle = {ECCV Workshops (Human-Centered Multimodal Intelligence in the Wild)},
  year      = {2026}
}
```
