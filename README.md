# SVG Logo LoRA Part B

This folder contains the Part B working files.

## What is ready

- Dataset cloned into `logo-detailed-prompt/`.
- First reward implementation in `student_kit/reward.py`.
- Reward sanity-check script in `student_kit/eval_reward_on_jsonl.py`.
- Training config starting point in `train_config.yaml`.
- Report template in `report.md`.
- Empty `adapter/` folder for the final LoRA files.

## Next steps

1. Download Gemma 3 270M from ModelScope into `gemma3-270m/`.
2. Run LoRA training with `train_config.yaml`.
3. Generate base-model and LoRA-model outputs on `valid.jsonl`.
4. Score both outputs with `student_kit/reward.py`.
5. Fill `results.json` and `report.md`.
