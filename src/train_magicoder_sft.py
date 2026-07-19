from __future__ import annotations

import argparse
import ast
import os
import random
import re
import warnings
from pathlib import Path

import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from src.train_mbpp_sft import (
    _docstring_prompt,
    _first_function,
    _looks_like_humaneval_leak,
    _plain_prompt,
    build_mbpp_examples,
)

_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.IGNORECASE | re.DOTALL)
_BAD_CONTEXT_MARKERS = (
    "django",
    "flask",
    "tensorflow",
    "torch.",
    "matplotlib",
    "pandas",
    "numpy",
    "requests.",
    "selenium",
    "open(",
    "argparse",
    "click.",
    "subprocess",
    "socket",
    "textio",
    "file object",
    "output file",
    "input file",
    "read file",
    "write file",
)
_BAD_PROMPT_MARKERS = (
    "web server",
    "web application",
    "database",
    "api endpoint",
    "file object",
    "input file",
    "output file",
    "command-line",
    "command line",
    "script",
    "django",
    "flask",
    "selenium",
    "provided code snippet",
    "global variable",
    "server",
    "connection",
    "url",
    "username",
    "password",
    "module",
    "project",
)


def _clean_problem(text: str) -> str:
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"(?is)the function signature is:.*", "", text)
    text = re.sub(r"(?is)you are provided with a code snippet.*?\n\n", "", text)
    task_match = re.search(r"(?is)(your task is to .*|write a function .*|implement a function .*)", text)
    if task_match is not None:
        text = task_match.group(1)
    text = re.sub(r"(?is)the code snippet provided.*?\.\s*", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text[:1400].rstrip()


def _strip_markdown_code(text: str) -> str:
    match = _FENCE_RE.search(text)
    if match is not None:
        return match.group(1).strip()
    return text.strip()


def _is_runnable_function_code(code: str) -> bool:
    if "def " not in code:
        return False
    lowered = code.lower()
    if any(marker in lowered for marker in _BAD_CONTEXT_MARKERS):
        return False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            module = ast.parse(code)
    except SyntaxError:
        return False
    functions = [node for node in module.body if isinstance(node, ast.FunctionDef)]
    if not functions:
        return False
    for node in ast.walk(module):
        if isinstance(node, (ast.AsyncFunctionDef, ast.ClassDef)):
            return False
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"input", "exec", "eval"}:
            return False
    return True


def _signature_key(code: str) -> str:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        parsed = _first_function(code)
    return parsed[0] if parsed is not None else code[:120]


def build_magicoder_oss_examples(seed: int, limit: int, mode: str) -> Dataset:
    if limit <= 0:
        return Dataset.from_list([])
    dataset = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train")
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in dataset:
        if (item.get("lang") or "").lower() != "python":
            continue
        raw_problem = (item.get("problem") or "").strip()
        problem = _clean_problem(raw_problem)
        solution = _strip_markdown_code(item.get("solution") or "")
        combined = f"{raw_problem}\n{solution}"
        if not problem or not solution or _looks_like_humaneval_leak(combined):
            continue
        if any(marker in combined.lower() for marker in _BAD_PROMPT_MARKERS):
            continue
        if not _is_runnable_function_code(solution):
            continue
        key = _signature_key(solution)
        if key in seen:
            continue
        seen.add(key)
        if mode == "native":
            rows.append({"text": _plain_prompt(problem, []) + solution + "\n"})
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            parsed = _first_function(solution)
        if parsed is None:
            continue
        header, body = parsed
        if len(body.splitlines()) > 90:
            continue
        rows.append({"text": _docstring_prompt(header, problem, []) + body + "\n"})
    random.Random(seed).shuffle(rows)
    return Dataset.from_list(rows[:limit])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--work-dir", default="outputs/magicoder_sft")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--magicoder-limit", type=int, default=2500)
    parser.add_argument("--magicoder-mode", choices=("completion", "native"), default="completion")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=768)
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
    magicoder = build_magicoder_oss_examples(args.seed, args.magicoder_limit, args.magicoder_mode)
    train_dataset = concatenate_datasets([mbpp, magicoder]).shuffle(seed=args.seed)
    print(f"training examples: mbpp={len(mbpp)} magicoder={len(magicoder)} total={len(train_dataset)}")

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
        max_length=args.max_length,
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
