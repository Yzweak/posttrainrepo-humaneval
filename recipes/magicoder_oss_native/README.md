# Magicoder OSS Native SFT

## Method

TRL `SFTTrainer` trains the same Qwen2.5-1.5B-Instruct LoRA setup as the MBPP anchor, then merges the adapter into a full checkpoint. This is the source-quality control for the Magicoder direction: MBPP sanitized remains unchanged and the supplemental examples come from the same filtered Magicoder OSS-Instruct Python subset, but stay in native instruction-to-solution format instead of being rewritten as function completions.

## Data provenance

- `google-research-datasets/mbpp`, config `sanitized`, all splits (`train`, `validation`, `test`, `prompt`): converted into both plain function-instruction examples and HumanEval-like function/docstring completion examples.
- `ise-uiuc/Magicoder-OSS-Instruct-75K`, split `train`: Python-language rows only, filtered for parseable Python code with top-level function definitions, no explicit HumanEval/openai_humaneval mentions, no classes/async functions, no `input`/`exec`/`eval`, and no obvious project-context/external-library markers. The selected supplement is shuffled with seed `13` and capped at 2500 examples.
- No HumanEval problems or solutions are loaded or used for training.

## Prompt / response template

MBPP uses the existing anchor templates. Magicoder rows use native instruction-to-solution form:

```python
# Write a Python function for the following task.
# <cleaned Magicoder problem text>
<parseable Python function solution>
```

## Hyperparameters

- Base model: `$MODEL_PATH` (`Qwen/Qwen2.5-1.5B-Instruct`), bf16.
- LoRA: `r=64`, `alpha=128`, `dropout=0.05`, target Qwen attention and MLP projection modules.
- SFT: `max_steps=200`, `max_length=768`, `packing=False`, batch size `8`, gradient accumulation `4`, effective batch `32`.
- Optimizer/schedule: `adamw_torch_fused`, `lr=1e-4`, cosine schedule, `warmup_ratio=0.05`, seed `13`.

## Results

- Local full HumanEval pass@1: `0.4207`.
- This native-format control underperformed the completion-format local run (`0.4390`) and the prior low-LR mixed-data anchor (`0.4451`).
