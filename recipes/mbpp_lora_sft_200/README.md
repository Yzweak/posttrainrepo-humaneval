# MBPP LoRA SFT 200 Steps

## Method

TRL `SFTTrainer` trains a LoRA adapter on the same MBPP plus Python-instruction function-generation mix as the previous recipes, then merges it into a full checkpoint. This variant uses `lr=1e-4` and `max_steps=200` to test a slightly longer conservative schedule than `mbpp_lora_sft_low_lr`.

## Data provenance

- `google-research-datasets/mbpp`, config `sanitized`, all splits (`train`, `validation`, `test`, `prompt`): converted into HumanEval-like function/docstring prompts plus canonical code solutions.
- `iamtarun/python_code_instructions_18k_alpaca`, split `train`: first 2500 shuffled examples containing Python functions, filtered to remove examples mentioning `HumanEval`/`openai_humaneval`.
- No HumanEval problems or solutions are loaded or used for training.

## Prompt / response template

```python
def function_name(args):
    """
    <MBPP task text>

    >>> <assertion without leading assert>
    """
    <solution body>
```

```python
# Write a Python function for the following task.
# <instruction and optional input>
<python function solution>
```

## Hyperparameters

- Base model: `$MODEL_PATH` (`Qwen/Qwen2.5-1.5B-Instruct`), bf16.
- LoRA: `r=64`, `alpha=128`, `dropout=0.05`, target Qwen attention and MLP projection modules.
- SFT: `max_steps=200`, `max_length=768`, `packing=False`, batch size `8`, gradient accumulation `4`, effective batch `32`.
- Optimizer/schedule: `adamw_torch_fused`, `lr=1e-4`, cosine schedule, `warmup_ratio=0.05`, seed `13`.

## Results

- Local base model: HumanEval pass@1 `0.3780`.
- Judge current best before this recipe: `0.426829`.
- Local this recipe: HumanEval pass@1 `0.4390`.
