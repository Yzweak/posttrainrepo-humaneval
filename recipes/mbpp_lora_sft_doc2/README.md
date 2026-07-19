# MBPP LoRA SFT Docstring Repeat 2

## Method

This is a Lead A warm-start variant. It preserves the mixed MBPP plus filtered Python-instruction SFT recipe, but repeats each MBPP docstring/function-body example once so the SFT objective is slightly more concentrated on the HumanEval-like completion format.

## Data provenance

- `google-research-datasets/mbpp`, config `sanitized`, all splits: plain prompts once and docstring prompts twice.
- `iamtarun/python_code_instructions_18k_alpaca`, split `train`: 2500 shuffled function-generation examples filtered for `HumanEval` / `openai_humaneval` mentions.
- No HumanEval problems or solutions are loaded or used for training.

## Hyperparameters

- Base model: `$MODEL_PATH` (`Qwen/Qwen2.5-1.5B-Instruct`), bf16.
- LoRA: `r=64`, `alpha=128`, `dropout=0.05`, target Qwen attention and MLP projection modules.
- SFT: `max_steps=180`, `lr=1e-4`, batch size `8`, gradient accumulation `4`, max length `768`.
- Final artifact: LoRA adapter merged into a full bf16 checkpoint in `$OUTPUT_DIR`.
