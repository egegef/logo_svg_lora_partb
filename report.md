# Part B Report

## Goal

The goal was to fine-tune Gemma 3 270M with LoRA so that it generates SVG logos from detailed visual prompts. The target was relative improvement over the 270M base model, not matching Sonnet-level visual quality.

## Reward Design

My reward is a programmatic proxy with six parts:

- Validity: parseable XML, `<svg>` root, correct namespace, `viewBox="0 0 256 256"`, and no unsafe tags.
- Structure: a reasonable number of vector elements such as `path`, `circle`, `rect`, `polygon`, `line`, and `ellipse`.
- Geometry: numeric coordinates should be finite and mostly near the 256 by 256 canvas.
- Palette: the output should use fill or stroke and a small number of colors.
- Prompt alignment: simple checks for obvious shape and color concepts mentioned in the prompt.
- Degeneration penalties: empty output, extra prose, unsafe content, repeated-character runs, and malformed SVG.

I added low partial credit for malformed SVGs that still contain a plausible SVG prefix, correct namespace, viewBox, shape tags, and colors. This is capped at 0.35, so invalid XML cannot receive a high score. The reason is that the base model and LoRA model often fail differently: the base model tends to produce prose or invalid namespaces, while the LoRA model often starts a plausible SVG but fails to close it. The reward should expose that difference while still making validity failure obvious.

## Training Setup

Model: `google/gemma-3-270m-it`, downloaded from ModelScope.

Source training data: `logo-detailed-prompt/train.jsonl`, 219 examples.

Source validation data: `logo-detailed-prompt/valid.jsonl`, 17 examples.

Final training data: `closure_data/train_closure.jsonl`, generated from the source prompts by `student_kit/make_closure_dataset.py`.

Final validation data: `closure_data/valid_closure.jsonl`, generated the same way.

I first trained on the original Sonnet SVG targets. That improved SVG-like structure but still failed XML parsing because the small model often did not close long SVG documents. I then used a closure curriculum: short, valid SVG targets that preserve prompt colors and simple shape concepts while strongly teaching `</svg>` completion. The submitted adapter is from this second training run.

LoRA config:

| Setting | Value |
|---|---:|
| Rank | 8 |
| Alpha | 16 |
| Dropout | 0.05 |
| Learning rate | 2e-4 |
| Epochs | 8 |
| Batch size | 1 |
| Gradient accumulation | 8 |
| Max length | 1024 |
| Seed | 43 |

Training used the original chat template and masked the prompt tokens so loss was computed only on the assistant SVG portion.

## Training Loss

The logged checkpoint summaries visible during training were:

| Epoch | Train loss | Validation loss |
|---:|---:|---:|
| 1 | 1.0906 | 0.4030 |
| 3 | 0.0306 | 0.0223 |
| 4 | 0.0173 | 0.0182 |
| 5 | 0.0132 | 0.0185 |
| 6 | 0.0117 | 0.0159 |
| 8 | 0.0098 | 0.0155 |

The adapter from epoch 8 was saved as the final adapter.

## Self-Evaluation

Decoding used greedy generation with `max_new_tokens=768`. The Gemma `<end_of_turn>` token was included as a stopping token.

| Model | Mean reward | Validity | Structure | Geometry | Palette | Prompt alignment | Penalty |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base Gemma 3 270M | 0.0937 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.7500 |
| LoRA adapter | 0.8869 | 1.0000 | 0.9592 | 1.0000 | 1.0000 | 0.4213 | 0.0000 |

Delta mean reward: `+0.7931`.

The submitted LoRA adapter now receives strict XML/SVG component credit. Unlike
the first original-target run, the final closure-curriculum run produced
parseable, closed SVGs on all 17 validation examples.

## Qualitative Findings

The base model often produced malformed SVG-like text with the wrong namespace, prose, markdown fences, or repeated attributes. Example start:

```xml
<svg xmlns="http://www.w3.org/svg" viewBox="0 0 256 256" fill="none" ...
```

The final LoRA model reliably starts with the correct SVG namespace and viewBox, uses valid SVG elements, and closes the document:

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256"><defs>...
```

The closure curriculum makes the outputs simpler than the original Sonnet targets, so the main remaining weakness is visual richness and fine prompt fidelity rather than validity.

## Analysis

The training loss and validation loss both improved, and the reward shows a clear relative gain over the base model. The improvement is now not only structural: the model learned to produce complete, parseable SVG documents.

The strongest result is validity: the LoRA model reaches `1.0000` validity, geometry, and palette on the self-evaluation set. Prompt alignment is lower (`0.4213`) because the generated curriculum intentionally uses compact template-like SVGs, so it does not capture every detailed style instruction.

The tradeoff is deliberate. For this small model, shorter closed SVGs satisfy the most important technical requirement better than long Sonnet-like targets. A next experiment could mix the closure curriculum with a small number of richer original SVGs after the model has learned to stop correctly.

## Reproducibility

The submitted repository contains:

- `adapter/adapter_config.json`
- `adapter/adapter_model.safetensors`
- `reward.py`
- `train_config.yaml`
- `results.json`
- `student_kit/train_lora_peft.py`
- `student_kit/eval_self.py`
- `student_kit/make_closure_dataset.py`
