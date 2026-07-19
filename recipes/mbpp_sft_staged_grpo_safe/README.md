# MBPP SFT + Conservative Staged GRPO

## Method

This is a lower-drift staged-GRPO variant. It uses the same mixed-data SFT warm-start and the same staged executable MBPP reward as `mbpp_sft_staged_grpo`, but reduces the GRPO update count and learning rate to test whether small execution-feedback nudges transfer better to HumanEval-style prompts.

## Data provenance

- `google-research-datasets/mbpp`, config `sanitized`, all splits: SFT examples plus GRPO prompts/rewards.
- `iamtarun/python_code_instructions_18k_alpaca`, split `train`: 2500 filtered function-generation examples for mixed SFT.
- No HumanEval problems or solutions are loaded or used for training.

## Reward

Staged reward: syntax penalty, partial credit for compilable expected-entry-point functions, fractional assertion credit, and a bonus for passing all MBPP tests.

## Hyperparameters

- SFT warm-start: `max_steps=180`, `lr=1e-4`, batch size `8`, gradient accumulation `4`, max length `768`.
- GRPO: `max_steps=30`, `lr=2e-6`, `num_generations=4`, completion length `192`, `temperature=0.7`, `beta=0.02`.
- Final artifact: LoRA adapter merged into a full bf16 checkpoint in `$OUTPUT_DIR`.
