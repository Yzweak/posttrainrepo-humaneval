# MBPP LoRA SFT

## Method

TRL `SFTTrainer` trains a LoRA adapter on Python function-generation examples, then merges the adapter into the base Qwen2.5-1.5B-Instruct weights and writes a full checkpoint to `$OUTPUT_DIR`.

## Data provenance

- `google-research-datasets/mbpp`, config `sanitized`, all splits (`train`, `validation`, `test`, `prompt`): converted into HumanEval-like function/docstring prompts plus canonical code solutions.
- `iamtarun/python_code_instructions_18k_alpaca`, split `train`: first 2500 shuffled examples containing Python functions, filtered to remove examples mentioning `HumanEval`/`openai_humaneval`.
- No HumanEval problems or solutions are loaded or used.

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
- SFT: `max_steps=260`, `max_length=768`, `packing=False`, batch size `8`, gradient accumulation `4`, effective batch `32`.
- Optimizer/schedule: `adamw_torch_fused`, `lr=2e-4`, cosine schedule, `warmup_ratio=0.05`, seed `13`.

## Results

Pending judge feedback. Record `job_id`, base pass@1, candidate pass@1, and delta after submission.
