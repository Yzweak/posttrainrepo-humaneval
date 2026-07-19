from __future__ import annotations

import argparse
import ast
import contextlib
import io
import os
import random
import signal
from pathlib import Path
from types import FrameType
from typing import Any

import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer

from src.train_mbpp_sft import (
    _docstring_prompt,
    _first_function,
    build_mbpp_examples,
    build_python_instruction_examples,
)


class TimeoutError(Exception):
    pass


def _timeout_handler(signum: int, frame: FrameType | None) -> None:
    raise TimeoutError("execution timed out")


def _function_name(header: str) -> str | None:
    try:
        module = ast.parse(header + "\n    pass\n")
    except SyntaxError:
        return None
    node = module.body[0] if module.body else None
    return node.name if isinstance(node, ast.FunctionDef) else None


def build_mbpp_reward_prompts(seed: int, limit: int = 0) -> Dataset:
    dataset = load_dataset("google-research-datasets/mbpp", "sanitized")
    rows: list[dict[str, Any]] = []
    for split in ("train", "validation", "test", "prompt"):
        for item in dataset[split]:
            parsed = _first_function(item["code"].strip())
            if parsed is None:
                continue
            header, _ = parsed
            entry_point = _function_name(header)
            tests = [test.strip() for test in item.get("test_list") or [] if test.strip()]
            if entry_point is None or not tests:
                continue
            rows.append(
                {
                    "prompt": _docstring_prompt(header, item["prompt"], tests),
                    "test_imports": list(item.get("test_imports") or []),
                    "test_list": tests,
                    "entry_point": entry_point,
                    "task_id": int(item["task_id"]),
                }
            )
    random.Random(seed).shuffle(rows)
    if limit > 0:
        rows = rows[:limit]
    return Dataset.from_list(rows)


def _strip_completion(completion: str) -> str:
    text = completion.split("<|endoftext|>", 1)[0]
    text = text.split("<|im_end|>", 1)[0]
    if "```" in text:
        text = text.replace("```python", "```").split("```", 1)[0]
    return text.rstrip() + "\n"


def _safe_exec(code: str, namespace: dict[str, Any], timeout: float) -> bool:
    previous = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            exec(code, namespace, namespace)
        return True
    except Exception:
        return False
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


def staged_execution_reward(
    prompts: list[str],
    completions: list[str],
    test_imports: list[list[str]],
    test_list: list[list[str]],
    entry_point: list[str],
    **_: Any,
) -> list[float]:
    rewards: list[float] = []
    for prompt, completion, imports, tests, name in zip(
        prompts, completions, test_imports, test_list, entry_point, strict=True
    ):
        source = prompt + _strip_completion(completion)
        try:
            ast.parse(source)
        except SyntaxError:
            rewards.append(-0.35)
            continue

        namespace: dict[str, Any] = {}
        setup = "\n".join(imports or [])
        if setup and not _safe_exec(setup, namespace, 0.5):
            rewards.append(-0.2)
            continue
        if not _safe_exec(source, namespace, 0.8) or name not in namespace:
            rewards.append(0.05)
            continue

        passed = 0
        for test in tests:
            if _safe_exec(test, namespace, 0.5):
                passed += 1
        fraction = passed / max(1, len(tests))
        reward = 0.15 + 0.85 * fraction
        if passed == len(tests):
            reward += 0.5
        rewards.append(reward)
    return rewards


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--work-dir", default="outputs/mbpp_sft_grpo")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--instruction-limit", type=int, default=2500)
    parser.add_argument("--sft-steps", type=int, default=180)
    parser.add_argument("--sft-learning-rate", type=float, default=1e-4)
    parser.add_argument("--grpo-steps", type=int, default=60)
    parser.add_argument("--grpo-learning-rate", type=float, default=5e-6)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--reward-limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    work_dir = Path(args.work_dir)
    adapter_dir = work_dir / "adapter"
    output_dir = Path(args.output_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    mbpp = build_mbpp_examples(args.seed)
    instruction = build_python_instruction_examples(args.seed, args.instruction_limit)
    sft_dataset = concatenate_datasets([mbpp, instruction]).shuffle(seed=args.seed)
    grpo_dataset = build_mbpp_reward_prompts(args.seed + 1, args.reward_limit)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    sft_args = SFTConfig(
        output_dir=str(work_dir / "sft_trainer"),
        dataset_text_field="text",
        max_length=768,
        packing=False,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        max_steps=args.sft_steps,
        learning_rate=args.sft_learning_rate,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
    )
    sft_trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=sft_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    sft_trainer.train()
    model = sft_trainer.model
    del sft_trainer
    torch.cuda.empty_cache()

    grpo_args = GRPOConfig(
        output_dir=str(work_dir / "grpo_trainer"),
        max_completion_length=192,
        per_device_train_batch_size=args.num_generations,
        gradient_accumulation_steps=2,
        max_steps=args.grpo_steps,
        learning_rate=args.grpo_learning_rate,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        num_generations=args.num_generations,
        temperature=0.7,
        top_p=0.95,
        beta=0.02,
        loss_type="dr_grpo",
    )
    grpo_trainer = GRPOTrainer(
        model=model,
        reward_funcs=staged_execution_reward,
        args=grpo_args,
        train_dataset=grpo_dataset,
        processing_class=tokenizer,
    )
    grpo_trainer.train()
    grpo_trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    del grpo_trainer, model
    torch.cuda.empty_cache()

    base = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="cuda:0",
    )
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    merged.save_pretrained(output_dir, safe_serialization=True, max_shard_size="4GB")
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
