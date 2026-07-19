# MBPP SFT + Binary GRPO

## Method

This recipe is the sparse-control branch for the assigned SFT + execution-reward GRPO direction. It keeps the mixed MBPP/Python-instruction SFT warm-start and then applies TRL GRPO on MBPP prompts with an all-or-nothing executable reward.

## Data provenance

- `google-research-datasets/mbpp`, config `sanitized`, all splits: SFT examples plus GRPO prompts/rewards.
- `iamtarun/python_code_instructions_18k_alpaca`, split `train`: 2500 filtered function-generation examples for the SFT warm-start.
- No HumanEval problems or solutions are loaded or used for training.

## Reward

The binary reward returns `1.0` only when the generated function compiles, defines the expected entry point, and passes every MBPP assertion. Any partial or failing solution receives `0.0`. This is intended as a direct control against the denser staged reward.

## Hyperparameters

- SFT warm-start: `max_steps=180`, `lr=1e-4`, batch size `8`, gradient accumulation `4`, max length `768`.
- GRPO: `max_steps=80`, `lr=1e-5`, `num_generations=4`, completion length `192`, `temperature=0.7`, `beta=0.02`.
- Final artifact: LoRA adapter merged into a full bf16 checkpoint in `$OUTPUT_DIR`.
