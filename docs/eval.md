# Evaluation

## In one line

The judge scores your checkpoint with lm-eval's **HumanEval** task, metric
**pass@1**, over all **164 problems**. Your score is the improvement over the
untrained base model.

## How the judge evaluates

After you run `bash submit.sh <recipe>`, the judge, in the background:

1. `git clone`s your repo (committed files only) and runs
   `recipes/<recipe>/run.sh` from a clean copy
2. run.sh must leave a checkpoint in `$OUTPUT_DIR` loadable by `from_pretrained()`
3. Evaluates that checkpoint with lm-eval:

   ```bash
   lm_eval run --model hf \
     --model_args pretrained=$OUTPUT_DIR,dtype=bfloat16,trust_remote_code=True \
     --tasks humaneval --limit 164 \
     --confirm_run_unsafe_code
   ```

4. Reads the `pass@1,create_test` metric
5. **Score = your checkpoint's pass@1 − base model's pass@1** (the baseline is
   evaluated once and cached)

## What you need to know

- **Metric**: `pass@1` — one sample generated per problem; it counts only if the
  generated code passes all of the problem's unit tests.
- **Coverage**: all 164 problems (the full HumanEval set, not a subset — a subset
  would bias the baseline).
- **Code is executed**: the harness actually runs the generated code
  (`--confirm_run_unsafe_code`), so the model must emit a **runnable Python
  function body**, not prose.
- **Optimize for code generation**: your training objective should push toward
  "write a function that passes the unit tests," not general chat quality.

## Submission is asynchronous

`submit.sh` returns a `job_id` immediately; evaluation runs in the background
(~30–60 min). Poll with `check.sh <job_id>` — it returns at once, so if it says
`running`, do other work and check again later. Do not block on `check.sh`, and
do not background `submit.sh` with `&`.

## Benchmark data must not enter training

HumanEval problems and solutions must **never** appear in your training corpus.
The judge is a hidden evaluation; you cannot read the problems. Train on other
code data (e.g. MBPP, open-source code, synthetic data).
