# MBPP LoRA SFT Low LR

## Method

TRL `SFTTrainer` trains a LoRA adapter on Python function-generation examples and merges it into the base Qwen2.5-1.5B-Instruct checkpoint. This is a conservative variant of `mbpp_lora_sft`: same data mix, fewer steps, and lower learning rate.

## Data provenance

- `google-research-datasets/mbpp`, config `sanitized`, all splits (`train`, `validation`, `test`, `prompt`): converted into HumanEval-like function/docstring prompts plus canonical code solutions.
- `iamtarun/python_code_instructions_18k_alpaca`, split `train`: first 2500 shuffled examples containing Python functions, filtered to remove examples mentioning `HumanEval`/`openai_humaneval`.
- No HumanEval problems or solutions are loaded or used for training.

## Prompt / response template

MBPP canonical template:

```python
def function_name(args):
    """
    <MBPP task text>

    >>> <assertion without leading assert>
    """
    <solution body>
```

Additional instruction template:

```python
# Write a Python function for the following task.
# <instruction and optional input>
<python function solution>
```

## Hyperparameters

- Base model: `$MODEL_PATH` (`Qwen/Qwen2.5-1.5B-Instruct`), bf16.
- LoRA: `r=64`, `alpha=128`, `dropout=0.05`, target Qwen attention and MLP projection modules.
- SFT: `max_steps=180`, `max_length=768`, `packing=False`, batch size `8`, gradient accumulation `4`, effective batch `32`.
- Optimizer/schedule: `adamw_torch_fused`, `lr=1e-4`, cosine schedule, `warmup_ratio=0.05`, seed `13`.

## Results

- Local base model: HumanEval pass@1 `0.3780`.
- Local `mbpp_lora_sft`: HumanEval pass@1 `0.4207`.
- Judge `mbpp_lora_sft`: job `a1a64daa-0f92-4050-8ee7-c4ee97b52563`, candidate `0.426829`, baseline `0.3780`, delta `+0.0488`.
- Local this recipe: HumanEval pass@1 `0.4451`.

## What mattered / what failed

- Keeping the 2500-example instruction mix helped; MBPP-only overfit and scored `0.3232` locally.
- Expanding to 8000 instruction examples with `max_steps=360`, `lr=1e-4` scored only `0.3659` locally.
