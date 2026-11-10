# Results and measurements

Reference for the write-up. Every number below was re-checked against the run
artifacts (`config.yaml`, `output.log`, `humaneval_samples*_eval_results.json`) or
re-measured on the corpus; anything derived or sampled says so. Nothing here is
estimated from a plan.

Metric: HumanEval+ mini, 164 problems, pass@1. `base` is the number to quote.
`plus` adds extra tests. `nodoc` is the same weights with the docstring removed
from the prompt.

---

## 1. HumanEval

| # | run | model total / active | corpus | passes | base | plus | nodoc |
|---|-----|----------------------|--------|--------|------|------|-------|
| 1 | `gemma4_moe__python_edu_gpu` | 270.7M / 228.2M | strict | 5 | 20/164 | 14/164 | — |
| 2 | same, shorter | 270.7M / 228.2M | strict | 2 | 13/164 | 10/164 | — |
| 3 | `config.ablation.yaml` | 155.8M / 131.0M | strict | 9 | 18/164 | 15/164 | — |
| 4 | `config.relaxed.yaml` | 155.8M / 131.0M | relaxed | 4 | **31/164** | 25/164 | 24/164 |
| 5 | run 4, docstrings stripped | same weights | — | — | 24/164 | 19/164 | — |
| 6 | `config.relaxed_15m.yaml` | 15.4M / 13.1M | relaxed | 4 | 19/164 | 15/164 | — |
| 7 | `config.file_15m.yaml` | 15.4M / 13.1M | whole file, 1024 | 1 | 0/164 | 0/164 | 0/164* |
| 8 | run 7 + `epochs: 2` | 15.4M / 13.1M | whole file, 1024 | 2 | 0/164 | 0/164 | 0/164* |
| 9 | `config.file_15m_4k.yaml` | 15.4M / 13.1M | whole file, 4096 | 1 | 0/164 | 0/164 | 0/164* |
| 10 | `config.file_dense_15m.yaml` | 13.051M / 13.051M | whole file, 1024 | 1 | 0/164 | 0/164 | 0/164* |

\* The file-level `nodoc` zero is a harness artifact, not a docstring-reliance
measurement. `evalplus` builds `solution = prompt + completion` with the
*docstring* prompt even for the `nodoc` pass, and these models emit tab-indented
bodies, so 160/164 die with `inconsistent use of tabs and spaces in indentation`.
Spliced onto the actually-stripped prompt, 146/164 (run 10) and 148/164 (run 7)
compile; they still fail the tests. The function-level `nodoc` (24/164, run 5) is
unaffected because that model emits spaces and 159/164 compile.

Active parameters are total minus one idle expert per layer (`top_k: 2` of 3);
13.1M is 13.07M rounded. The dense control is 13.051M because it has one FFN of
width 1536 (768 shared + 2 x 384) instead of the two parallel FFN paths.

## 2. What moved the number

Paired McNemar over per-task pass/fail, exact two-sided. The older independent
two-proportion z-tests treated the runs as separate samples; they share the same
164 problems, and pairing changes the answer.

| comparison | discordant (only A / only B) | exact p |
|---|---|---|
| 10x params, same data (19 vs 31) | 1 / 13 | 0.0018 |
| 3.6x data, same params (18 vs 31) | 2 / 15 | 0.0024 |
| docstring removed, same weights (31 vs 24) | 8 / 1 | 0.039 |
| function task vs file task (19 vs 0) | 19 / 0 | 3.8e-06 |

Read: **both** more unique data and more capacity moved the score once the corpus
was relaxed, and the docstring gap is borderline real rather than noise. The
earlier "capacity bought +0" result came from a sweep that stopped at 63.8M
active on strict data; it never extended down to 13.1M.

## 3. Function-level corpora

```
strict (docstring required)
  examples            752,420      train 601,936 (37,621 steps x 16)
  train tokens/pass   ~92.6M       mean 153.8 tokens
relaxed (require_docstring: false)
  examples          4,180,678      train 3,344,576 (52,259 steps x 64)
  train tokens/pass   ~336M        mean 100.1 tokens
  split               17.5-17.8% documented (~154 tokens), 82% bare (~89 tokens)
```

Splits are by function, so siblings of the same file can land in train and val.
HumanEval is external and unaffected.

## 4. File-level: four runs, all 0/164

| run | only change vs run 7 | train tokens | base |
|---|---|---|---|
| 7 | — | 1.29B | 0/164 |
| 8 | `epochs: 2` | 2.58B | 0/164 |
| 9 | `max_length: 4096`, batch 8 | 1.98B | 0/164 |
| 10 (dense) | MoE FFN replaced by one dense FFN | 1.29B | 0/164 |

Config diffs are exact: run 8 changes only `trainer.epochs`; run 9 changes the two
`max_length` fields and `batch_size`; run 10 changes the model and loss blocks
(dense cross-entropy has no router to balance) and nothing else.

Why the zero, from the saved generations:

```
base generations, docstring in the prompt
  run 7      164/164 empty   (EOS immediately after """ )
  run 10     164/164 empty
  run 9      162/164 empty
nodoc, docstring stripped
  run 7/10   160/164 write a tab-indented body, 160/164 IndentationError
             4 parse, all fail the tests
```

The model is not producing commented-out code. It stops at the docstring; when the
docstring is removed it writes a plausible body with the wrong indent. The dense
control produces the same counts, so this is not a routing effect.

Diagnostic probe (`probe_body_rate.py`, 99 identical test files cut at a closing
docstring, greedy, 128 tokens): run 7 writes a body 8% of the time, dense 9%.
The previously recorded 17%/10%/16% body rates have no surviving artifacts and
were produced by a script that no longer exists; do not quote them.

## 5. File corpus

Measured on the filtered cache and on 30k-file samples spread across all shards.

```
max_length 1024
  files                 5,590,668
  tokens/pass           1.61B      (mean 287.7)
  train tokens/pass     1.29B      (80/10/10)
  length                median 222, p75 392, p90 620, p99 958
  fit <=512             84% of files in the cache
  eligibility           72.6% of raw rows: 11.3% do not parse,
                        9.4% import outside the allowlist, 6.7% exceed 1024
  documented defs/file  0.112
  after a docstring     0.039% of tokens, ~0.63M positions/pass
max_length 4096
  files                 6,071,031
  train tokens/pass     1.98B      (mean 408.6)
```

The long files the 1024 cap drops (1025-2048 tokens) are 6% of rows but +31% tokens
and +41% after-docstring positions, at 0.72 documented defs/file instead of 0.112.
That is why run 9 raised the cap; it still produced 0/164.

Raw corpus: 7,677,467 rows, 11.985B characters, ~3.93B tokens in the 8192 BPE
(3.06 chars/token, 512 tokens/file; token counts from a 50k-row sample).

The self-containment filter is why file-level training exists at all: on 20k source
rows, 16,887 top-level candidates, 11,294 kept (66.9%), 5,593 dropped (33.1%)
because they reference a module-level name from their own file - i.e. the composed
code that calls helpers. Whole files keep it, plus imports and module order.

Imports and repo structure:

```
non-stdlib root imports   10.2% of parsed files (rejected by the allowlist)
relative imports           0.34%
files per repo             median 1, 57% single-file, p90 8, max 25,751
```

The corpus has almost no intact repos, so repo-level pretraining is not an option
with this data.

## 6. Cost

```
function 15M    4 passes, batch 64, 512     7.8 h    209,036 steps
function 156M   4 passes, batch 24, 512    30.4 h    measured
file 15M        1 pass,  batch 32, 1024     5.2 h    139,767 steps (5.7 h to eval)
file 15M        2 passes, batch 32, 1024   11.2 h    279,534 steps (11.5 h to eval)
file 15M 4096   1 pass,  batch 8,  4096    17.8 h    607,103 steps, 14,364 MB
dense 15M       1 pass,  batch 32, 1024     3.2 h    measured (fewer params to update)
```

## 7. Decisions

| decision | reason |
|---|---|
| `max_length: 512` for functions | mean 154 (strict) and 100 (relaxed) tokens; 1024 recovers very little |
| `max_length: 1024` for files | 84% of cache files fit; 512 would drop ~16%, mostly multi-function files with helpers |
| `require_docstring: false` | 5.6x more functions; the only change that moved HumanEval materially |
| lr 6e-4 everywhere | below the critical batch size; sqrt-scaling adds a variable with no evidence |
| 4 passes as the repetition boundary | up to ~4 passes is nearly free (Muennighoff); run 8 shows a 2nd file pass also buys nothing |
| do not use Chinchilla 20 tok/param | D is unique tokens; the function corpus supports only ~17M active at that rule |
| file-level as the comparison baseline | same token budget (~1.3B each), so the task is the only variable |

## 8. Known limitations

- **Sibling leakage.** Splits are by function, not by file. Validation loss is
  optimistic and `best.pt` selection is noisy; HumanEval is unaffected.
- **Quality dilution.** Dropping the docstring filter also removed an implicit
  "somebody cared about this code" signal; 82% of the corpus is undocumented.
- **The `nodoc` file numbers are invalid**, for the harness reason in section 1.
- **The 17%/10%/16% body rates cannot be reproduced.** Only run 7 and the dense run
  have probe artifacts, and they read 8-9%.
- **The in-training benchmark prompts with the first half of the file** and scores
  2 samples. `bench_body=0.00` in the logs is neither the docstring probe nor a
  signal; do not compare it to the table above.
- **Corpus totals are derived.** File/token totals are mean length x file count
  (30k-file sample, +/-3%); the raw char count is exact, the raw token count is a
  50k-row sample.

## 9. Open questions

1. **Where is the size knee?** 13.1M -> 19, 131M -> 31 on the same corpus, and the
   paired test says that gap is real. The next arm is ~54M active: `~28` means the
   knee is low, `22-25` means it is near 50M.
2. **Does a mixed objective fix file-level training?** Every schedule and corpus
   lever produced 0/164. The remaining options are file-level as stage 1 then
   fine-tune on `signature -> body`, or mixing a body-completion loss into the file
   stream. Only after that is a larger file-level arm worth running.
3. **Does inference-time context help?** Pasting imported definitions into the
   prompt tests whether the model uses context at all. If it ignores it, retrieval
   over files cannot work.
