# MBPP SFT + Staged GRPO

## Method

This recipe keeps the strongest conservative mixed-data MBPP SFT warm-start, then continues the same LoRA policy with TRL GRPO on executable MBPP feedback. It follows the assigned direction by comparing against the existing SFT family while adding a staged compiler / unit-test reward rather than switching to unrelated data scaling.

## Data provenance

- `google-research-datasets/mbpp`, config `sanitized`, all splits: used for mixed SFT examples and for GRPO prompts/rewards.
- `iamtarun/python_code_instructions_18k_alpaca`, split `train`: 2500 shuffled function-generation examples for the SFT mixture, filtered to remove `HumanEval` / `openai_humaneval` mentions.
- No HumanEval problems or solutions are loaded or used for training.

## Reward

GRPO samples four completions for each MBPP docstring-style function prompt. The reward is staged: syntax failures are penalized, compilable functions get partial credit, each passing MBPP assertion adds fractional credit, and passing all assertions receives a bonus. This is intended to preserve the HumanEval-like completion format while giving denser feedback than all-or-nothing execution reward.

## Hyperparameters

- Base model: `$MODEL_PATH` (`Qwen/Qwen2.5-1.5B-Instruct`), bf16.
- LoRA: `r=64`, `alpha=128`, `dropout=0.05`, target Qwen attention and MLP projection modules.
- SFT warm-start: `max_steps=180`, `lr=1e-4`, batch size `8`, gradient accumulation `4`, max length `768`.
- GRPO: `max_steps=60`, `lr=5e-6`, `num_generations=4`, completion length `192`, `temperature=0.7`, `beta=0.02`.
- Final artifact: LoRA adapter merged into a full bf16 checkpoint in `$OUTPUT_DIR`.
