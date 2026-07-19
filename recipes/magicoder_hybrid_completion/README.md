# Magicoder Hybrid Completion SFT

## Method

TRL `SFTTrainer` trains the same LoRA SFT setup and merges the adapter into a full checkpoint. This recipe keeps the archived mixed-data anchor partially intact while adding Magicoder: MBPP sanitized is unchanged, the supplemental budget is split between 1000 completion-aligned Magicoder OSS-Instruct examples and 1500 filtered `iamtarun/python_code_instructions_18k_alpaca` examples.

## Data provenance

- `google-research-datasets/mbpp`, config `sanitized`, all splits (`train`, `validation`, `test`, `prompt`): existing function-instruction and HumanEval-like completion templates.
- `ise-uiuc/Magicoder-OSS-Instruct-75K`, split `train`: Python rows filtered and rewritten with `src.train_magicoder_sft.build_magicoder_oss_examples`, shuffled with seed `13`, capped at 1000 examples.
- `iamtarun/python_code_instructions_18k_alpaca`, split `train`: existing simple function-containing filter from `src.train_mbpp_sft.build_python_instruction_examples`, shuffled with seed `13`, capped at 1500 examples.
- No HumanEval problems or solutions are loaded or used for training.

## Prompt / response template

Magicoder rows use function-completion/docstring alignment:

```python
def function_name(args):
    """
    <cleaned Magicoder problem text>
    """
    <solution body>
```

The MBPP and `iamtarun` rows use the anchor templates documented in `recipes/mbpp_lora_sft_low_lr`.

## Hyperparameters

- Base model: `$MODEL_PATH` (`Qwen/Qwen2.5-1.5B-Instruct`), bf16.
- LoRA: `r=64`, `alpha=128`, `dropout=0.05`, target Qwen attention and MLP projection modules.
- SFT: `max_steps=200`, `max_length=768`, `packing=False`, batch size `8`, gradient accumulation `4`, effective batch `32`.
- Optimizer/schedule: `adamw_torch_fused`, `lr=1e-4`, cosine schedule, `warmup_ratio=0.05`, seed `13`.

## Results

- Local full HumanEval pass@1: `0.4268`.
- This 1000 Magicoder / 1500 archived-supplement blend did not beat pure Magicoder completion locally (`0.4390`) or the archived low-LR anchor (`0.4451`).
