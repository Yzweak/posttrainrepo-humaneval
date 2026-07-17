# Architecture

## Directories

- `src/`: shared modules. Code used by multiple recipes goes here (data loaders,
  reward functions, LoRA merge utilities, etc.).
- `recipes/`: one self-contained snapshot per experiment. Each subdirectory has
  `run.sh` + `README.md`.
- `docs/`: the docs you're reading. They describe how the repo itself works.
- `cookbook/`: runnable snippets and gotchas.

## src/ conventions

Modules in `src/` are imported by a recipe's `run.sh`. In run.sh:

```bash
cd "$(dirname "$0")/../.."   # back to repo root
uv sync
uv run python src/train.py --model_path "$MODEL_PATH" --output_dir "$OUTPUT_DIR"
```

When adding a module, make sure it:
1. imports cleanly via `uv run python -c "import src.<module>"`
2. has no hard-coded paths — pass them via arguments / environment variables
