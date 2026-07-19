from __future__ import annotations

import argparse
import ast
import os
import random
from pathlib import Path

import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def _indent(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line if line.strip() else line for line in text.splitlines())


def _first_function(code: str) -> tuple[str, str] | None:
    try:
        module = ast.parse(code)
    except SyntaxError:
        return None
    for node in module.body:
        if isinstance(node, ast.FunctionDef):
            segment = ast.get_source_segment(code, node) or code
            lines = segment.splitlines()
            if not lines:
                return None
            header = lines[0].rstrip()
            body = "\n".join(lines[1:]).rstrip()
            if header.startswith("def ") and body:
                return header, body
    return None


def _docstring_prompt(header: str, task: str, tests: list[str]) -> str:
    examples = []
    for test in tests[:3]:
        example = test.strip()
        if example.startswith("assert "):
            example = example[len("assert ") :]
        examples.append(f"    >>> {example}")
    doc_lines = [f"    {task.strip()}"]
    if examples:
        doc_lines.append("")
        doc_lines.extend(examples)
    return f'{header}\n    """\n' + "\n".join(doc_lines) + '\n    """\n'


def _plain_prompt(task: str, tests: list[str]) -> str:
    lines = ["# Write a Python function for the following task.", f"# {task.strip()}"]
    if tests:
        lines.append("# The function should satisfy these checks:")
        lines.extend(f"# {test.strip()}" for test in tests[:3])
    return "\n".join(lines) + "\n"


def build_mbpp_examples(seed: int) -> Dataset:
    dataset = load_dataset("google-research-datasets/mbpp", "sanitized")
    rows: list[dict[str, str]] = []
    for split in ("train", "validation", "test", "prompt"):
        for item in dataset[split]:
            code = item["code"].strip()
            parsed = _first_function(code)
            tests = list(item.get("test_list") or [])
            task = item["prompt"]
            rows.append({"text": _plain_prompt(task, tests) + code + "\n"})
            if parsed is not None:
                header, body = parsed
                rows.append({"text": _docstring_prompt(header, task, tests) + body + "\n"})
    random.Random(seed).shuffle(rows)
    return Dataset.from_list(rows)


def _looks_like_humaneval_leak(text: str) -> bool:
    lowered = text.lower()
    return "humaneval" in lowered or "openai_humaneval" in lowered


def build_python_instruction_examples(seed: int, limit: int) -> Dataset:
    dataset = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")
    rows: list[dict[str, str]] = []
    for item in dataset:
        instruction = (item.get("instruction") or "").strip()
        inputs = (item.get("input") or "").strip()
        output = (item.get("output") or "").strip()
        if not instruction or not output:
            continue
        combined = instruction + "\n" + inputs + "\n" + output
        if _looks_like_humaneval_leak(combined):
            continue
        if "def " not in output:
            continue
        prompt = _plain_prompt(instruction + (f" Input: {inputs}" if inputs else ""), [])
        rows.append({"text": prompt + output.removeprefix("# Python code").strip() + "\n"})
    random.Random(seed).shuffle(rows)
    return Dataset.from_list(rows[:limit])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--work-dir", default="outputs/mbpp_sft")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--instruction-limit", type=int, default=2500)
    parser.add_argument("--max-steps", type=int, default=260)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
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
    tokenizer.padding_side = "right"

    mbpp = build_mbpp_examples(args.seed)
    instruction = build_python_instruction_examples(args.seed, args.instruction_limit)
    train_dataset = concatenate_datasets([mbpp, instruction]).shuffle(seed=args.seed)

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
    train_args = SFTConfig(
        output_dir=str(work_dir / "trainer"),
        dataset_text_field="text",
        max_length=768,
        packing=False,
        shuffle_dataset=True,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
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
    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    del trainer, model
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
