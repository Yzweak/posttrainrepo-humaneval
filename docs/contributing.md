# 怎么新建一个 recipe

1. 在 `recipes/` 下新建目录(用有意义的名字,不要用 cycle 编号):
   ```
   recipes/grpo_exec_reward/
     run.sh
     README.md
   ```

2. `run.sh` 必须满足 [README 里的契约](../README.md#runsh-契约)。

3. `README.md` 至少包含:
   - 方法简述(一句话)
   - 使用的框架/库
   - 关键超参
   - 结果(分数,如果已知)

4. 通用代码放 `src/`,recipe 里 import 它。别在 recipe 里重复造轮子。

5. CI 会自动检查:
   - `run.sh` 存在且语法正确(shellcheck)
   - `README.md` 存在且非空
   - `src/` 里的模块能正常 import
   - 代码通过 ruff lint + format
