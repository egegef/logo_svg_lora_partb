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

Training data: `logo-detailed-prompt/train.jsonl`, 219 examples.

Validation data: `logo-detailed-prompt/valid.jsonl`, 17 examples.

LoRA config:

| Setting | Value |
|---|---:|
| Rank | 8 |
| Alpha | 16 |
| Dropout | 0.05 |
| Learning rate | 1e-4 |
| Epochs | 6 |
| Batch size | 1 |
| Gradient accumulation | 8 |
| Max length | 4096 |
| Seed | 42 |

Training used the original chat template and masked the prompt tokens so loss was computed only on the assistant SVG portion.

## Training Loss

| Epoch | Train loss | Validation loss |
|---:|---:|---:|
| 1 | 1.3430 | 1.0033 |
| 2 | 0.8929 | 0.8437 |
| 3 | 0.7942 | 0.7905 |
| 4 | 0.7530 | 0.7689 |
| 5 | 0.7338 | 0.7603 |
| 6 | 0.7275 | 0.7592 |

The adapter from epoch 6 was saved as the final adapter.

## Self-Evaluation

Decoding used greedy generation with `max_new_tokens=768`. The Gemma `<end_of_turn>` token was included as a stopping token.

| Model | Mean reward | Valid XML score | Structure score | Palette score | Prompt alignment |
|---|---:|---:|---:|---:|---:|
| Base Gemma 3 270M | 0.0937 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| LoRA adapter | 0.3300 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

Delta mean reward: `+0.2363`.

## Qualitative Findings

The base model often produced malformed SVG-like text with the wrong namespace, prose, markdown fences, or repeated attributes. Example start:

```xml
<svg xmlns="http://www.w3.org/svg" viewBox="0 0 256 256" fill="none" ...
```

The LoRA model usually started with the correct SVG namespace and viewBox, and generated recognizable SVG components such as gradients, circles, strokes, and colors:

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256"><defs>...
```

However, the LoRA outputs still usually failed XML parsing because they did not close the SVG cleanly before the generation limit. This is the main remaining failure.

## Analysis

The training loss and validation loss both improved, and the reward shows a clear relative gain over the base model. The improvement is mostly structural: the LoRA model learned the expected SVG opening pattern, namespace, viewBox, and common logo elements.

The result is still weak in absolute terms. The valid XML score stayed at zero for both models, so the adapter did not yet solve the most important requirement: producing a complete closed SVG document. This is a Goodhart risk for my reward: the model can receive partial malformed-SVG credit without producing a valid final logo. I capped malformed-output credit to keep this visible.

The next experiment should focus on shorter, closed SVG targets or an explicit closing-SVG curriculum. Another useful change would be to train/evaluate with stronger stopping behavior and examples that teach `</svg><end_of_turn>` more reliably.

## Reproducibility

The submitted repository contains:

- `adapter/adapter_config.json`
- `adapter/adapter_model.safetensors`
- `reward.py`
- `train_config.yaml`
- `results.json`
- `student_kit/train_lora_peft.py`
- `student_kit/eval_self.py`
