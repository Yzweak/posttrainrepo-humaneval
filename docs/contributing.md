# How to add a recipe

1. Create a directory under `recipes/` (use a meaningful name, not a cycle
   number):
   ```
   recipes/grpo_exec_reward/
     run.sh
     README.md
   ```

2. `run.sh` must satisfy the [contract in the README](../README.md#runsh-contract).

3. `README.md` should include at least:
   - a one-line method summary
   - the frameworks / libraries used
   - key hyperparameters
   - results (score, if known)

4. Put shared code in `src/` and import it from the recipe. Don't reinvent the
   wheel inside a recipe.

5. CI checks automatically:
   - `run.sh` exists and is syntactically valid (shellcheck)
   - `README.md` exists and is non-empty
   - modules in `src/` import cleanly
   - code passes ruff lint + format
