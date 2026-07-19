# Magicoder OSS Completion SFT

## Method

TRL `SFTTrainer` trains the same Qwen2.5-1.5B-Instruct LoRA setup as the MBPP anchor, then merges the adapter into a full checkpoint. This recipe keeps MBPP sanitized as the stabilizing HumanEval-like base and replaces the prior `iamtarun/python_code_instructions_18k_alpaca` supplement with a same-budget Magicoder OSS-Instruct Python subset rewritten into function-completion/docstring form.

## Data provenance

- `google-research-datasets/mbpp`, config `sanitized`, all splits (`train`, `validation`, `test`, `prompt`): converted into both plain function-instruction examples and HumanEval-like function/docstring completion examples.
- `ise-uiuc/Magicoder-OSS-Instruct-75K`, split `train`: Python-language rows only, filtered for parseable Python code with top-level function definitions, no explicit HumanEval/openai_humaneval mentions, no classes/async functions, no `input`/`exec`/`eval`, and no obvious project-context/external-library markers. The selected supplement is shuffled with seed `13` and capped at 2500 examples.
- No HumanEval problems or solutions are loaded or used for training.

## Prompt / response template

MBPP uses the existing anchor templates. Magicoder rows use completion alignment:

```python
def function_name(args):
    """
    <Magicoder problem text>
    """
    <solution body>
```

## Hyperparameters

- Base model: `$MODEL_PATH` (`Qwen/Qwen2.5-1.5B-Instruct`), bf16.
- LoRA: `r=64`, `alpha=128`, `dropout=0.05`, target Qwen attention and MLP projection modules.
- SFT: `max_steps=200`, `max_length=768`, `packing=False`, batch size `8`, gradient accumulation `4`, effective batch `32`.
- Optimizer/schedule: `adamw_torch_fused`, `lr=1e-4`, cosine schedule, `warmup_ratio=0.05`, seed `13`.

## Results

- Local full HumanEval pass@1 with stricter context filtering: `0.4390`.
- First judge run before the stricter filter commit: job `6b8df670-faa5-4bf8-b794-2bbd19fd90a6`, candidate `0.408537`, baseline `0.3780`, delta `+0.0305`.
- Current stricter-filter recipe pending judge evaluation.
