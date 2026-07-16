# posttrainrepo

Post-training 实验仓库。fork 此模板,开始你的 post-training 实验。

## 目录结构

```
src/          通用模块(data loader, reward function, merge 工具...)
docs/         仓库文档(架构、评测、模型信息)
cookbook/      可跑的代码片段和踩坑记录
recipes/      每次实验的独立快照(累加不覆盖)
  <name>/
    run.sh    一键脚本:建环境 → 训练 → 产出 checkpoint
    README.md 方法、超参、结论、分数
```

## 怎么用

1. 阅读 `docs/` 了解评测机制和 base 模型
2. 翻阅 `recipes/` 看看已有实验(如果有)
3. 在 `recipes/` 下新建你的实验目录
4. 写 `run.sh`(必须满足下面的契约)+ `README.md`
5. 可以在 `src/` 里添加/复用通用工具模块

## run.sh 契约

```bash
MODEL_PATH=<base 权重> OUTPUT_DIR=<产出目录> bash recipes/<name>/run.sh
```

- **自包含**:从 `uv sync` 开始,到训练完成、checkpoint 落入 `$OUTPUT_DIR`
- **幂等**:在干净 clone 上运行,不依赖本地残留
- **退出码**:成功 0;失败非 0
- **产出**:`$OUTPUT_DIR/` 必须能被 `AutoModelForCausalLM.from_pretrained()` 直接加载
- **时间**:1 小时内完成

## 原则

- **调库不造轮子**:优先用 TRL / veRL / OpenRLHF 等成熟库,保持代码精简
- **src/ 是通用的**:放多个 recipe 共享的模块;实验特有的逻辑留在 recipe 内
- **recipe 是独立的**:每个 recipe 可以独立跑,互不依赖
