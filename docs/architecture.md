# 架构说明

## 目录

- `src/`:通用模块。多个 recipe 共享的代码放这里(data loader, reward function, LoRA merge 工具等)。
- `recipes/`:每次实验的独立快照。每个子目录有 `run.sh` + `README.md`。
- `docs/`:你正在读的文档。描述 repo 本身怎么用。
- `cookbook/`:可跑的代码片段和踩坑记录。

## src/ 使用约定

src/ 里的模块被 recipe 的 run.sh 引用。在 run.sh 里:

```bash
cd "$(dirname "$0")/../.."   # 回到 repo 根
uv sync
uv run python src/train.py --model_path "$MODEL_PATH" --output_dir "$OUTPUT_DIR"
```

新增模块时请确保:
1. 能被 `python -c "import src.<module>"` 正常导入
2. 没有硬编码路径,通过参数/环境变量传入
