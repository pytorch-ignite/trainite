# Results and measurements

Running record of what has been run, what was measured, and what it means. Every
number here came from a real run or a real measurement; anything estimated is
marked. Update this file whenever a run finishes.

This file is a reference, not a write-up: every result carries a short note saying
what changed relative to the run before it and what the number therefore shows.
Read the table and its notes top to bottom and the argument follows.

---

## 1. HumanEval results

HumanEval+ mini, 164 problems, pass@1. `base` = standard prompts, `plus` = base
plus extra tests, `nodoc` = same weights, prompts with the docstring removed.

| # | run | model (total / active) | corpus | tokens | HumanEval base | plus | plus retention |
|---|-----|------------------------|--------|--------|----------------|------|----------------|
| 1 | `gemma4_moe__python_edu_gpu` | 270.7M / 228.2M | strict, 5 ep | 467M | 20/164 (12.2%) | 14/164 (8.5%) | 70% |
| 2 | `gemma4_moe__python_edu_gpu` ×2 | 270.7M / 228.2M | strict, 2 ep | — | 13/164 | 10/164 | 77% |
| 3 | `config.ablation.yaml` | 155.8M / 131.0M | strict, 9 ep | 841M (best ≈607M) | 18/164 (11.0%) | 15/164 (9.1%) | 83% |
| 4 | `gemma4_moe__relaxed_h768` | 155.8M / 131.0M | relaxed, 4 ep | 1.20B | **31/164 (18.9%)** | **25/164 (15.2%)** | 81% |
| 5 | run 4, docstrings stripped | same weights | — | — | 24/164 (14.6%) | 19/164 (11.6%) | 79% |
| 6 | `config.relaxed_15m.yaml` | 15.4M / 13.1M | relaxed, 4 ep | 1.25B | 19/164 (11.6%) | 15/164 (9.2%) | 79% |
| 7 | `config.file_15m.yaml` | 15.4M / 13.1M | file-level, 1 ep | ~1.3B | **0/164 (0.0%)** | **0/164 (0.0%)** | — |
| 8 | run 7 config, `epochs: 2` | 15.4M / 13.1M | file-level, 2 ep | ~2.6B | **0/164 (0.0%)** | **0/164 (0.0%)** | — |

### Comparisons (two-proportion z-test, n = 164)

```
15M relaxed (19)  vs 156M relaxed (31)   same corpus, 10x params
  base  -7.3pp  z=-1.84  p=0.065
  plus  -6.1pp  z=-1.69  p=0.091

156M relaxed (31) vs 156M strict (18)    4x data, same params
  base  +7.9pp  z=+2.03  p=0.043
  plus  +6.1pp  z=+1.69  p=0.090

156M relaxed (31) vs 270M strict (20)    different corpus
  base  +6.7pp  z=+1.68  p=0.092
  plus  +7.9pp  z=+2.29  p=0.022
```

Wilson 95% intervals: 31/164 = 13.6–25.6%, 24/164 = 10.0–20.9%, 19/164 = 7.5–17.4%,
18/164 = 7.1–16.7%, 20/164 = 8.0–18.1%. ±2–3 solved is noise on 164 problems.

**Not yet done:** paired McNemar over `humaneval_samples*_eval_results.json`, which
is the right test because both runs are scored on the same 164 problems.

### Why each run exists

```
1  baseline. The "big model" everything else was measured against.
2  same model, fewer epochs — showed the val curve was still falling, so the
   5-epoch number was not a ceiling. Both early runs live in
   `gemma4_moe__python_edu_gpu/`; the `gemma4_moe__vm` runs are separate
   aborted attempts (1/164, humaneval_batch_size 4) and are not in the table.
3  equal-compute ablation against run 1: 156M x 9 ep == 270M x 5 ep in FLOPs.
   Same score. This is where "size doesn't matter" came from — and where it
   stopped being true once the data changed.
4  the data experiment. Only the corpus changed. This is the headline.
5  the honesty check. Same weights, docstring removed from the prompt.
6  the capacity experiment. Only the size changed. This is what broke hypothesis 1.
7  the task experiment. Whole files instead of functions, matched token budget.
   Scored 0/164 — see notes below.
8  the epochs test on run 7. Only `epochs: 1` -> `2` changed. Still 0/164, and the
   body rate did not move. That answer is in the notes below.
```

### Notes on run 7 (and run 8): the 0/164

```
not a broken model, not a bad checkpoint, not the truncation flag:
  - same script + function-level 15M weights -> 19/164, matching that run's own log
  - load path fine: test() loaded best.pt before the crash in the original run
  - truncation off, re-generated:
      82/164  new top-level def, often re-declaring the given function
      70/164  started with #  -> right code, commented out
      12/164  EOS as first token
       0/164  indented body
    commented code and a re-declared function fail either way, so the flag is not the cause

the real test: corpus files cut right after a function docstring, models asked to
continue (n=100, 100 fresh files, not the 20 used before):

    function-level 15M, 4 ep      88% correct indented body
    file-level     15M, 1 ep      17%
    file-level     15M, 2 ep      10%

  - same size, same tokenizer, same source -> capacity is not the limit
  - run 7 vs run 8 differs only in `epochs: 1` -> `2` (config diff), so repetition
    is the only variable, and it bought nothing: 17% -> 10%
  - 10/100 and 17/100 are within noise of each other (p~0.15); the point is that
    neither is anywhere near 88%
  - run 8's val_loss still improved, 1.3357 -> 1.2847 (ppl 3.80 -> 3.61), while the
    body rate went nowhere. The loss is being spent on the 99.96% of tokens that are
    not the after-docstring position
  - dominant failure stays the same in both runs: right code, commented out
    (run 7: 43%, run 8: 53%), then a fresh top-level def (~26%), then EOS
  - mid-file prefix (nested loop, Chinese comments): continues cleanly -> file
    structure was learned. Only one move was missed

what it never learned: indent and write the body after a closing """

why: that position is ~0.04% of file-level tokens vs 18% of function-level examples
     (~460x less signal on the move HumanEval tests)

verdict: epochs are ruled out. 2 passes over the same data moved val loss and did
not move the behavior. The objective has to change, not the schedule:
  - mix the function-level task into file training, or
  - file-level as stage 1, then fine-tune on `signature -> body`
```

---

## 2. Headline conclusions

**Data volume was the binding constraint, and it was fixed.** 4× unique data bought
+13 problems; 3.5× parameters bought +0.

**Capacity binds between 13.1M and 131M active.** The earlier sweep (63.8M → 228.2M
active, all on strict data) found no difference. Now:

```
13.1M active  ->  19/164
131.0M active ->  31/164     same corpus, same recipe, only size differs
```

The knee is unmeasured. This is the most valuable open number.

**Data and capacity substitute, at roughly this exchange rate:**

```
156M strict  (18)  ==  15M relaxed (19)   10x less capacity, 4x more data, +1
+13 from 4x data (at 156M)  ==  +12 from 10x params (at relaxed)
```

**Chinchilla's 20 tok/param is the wrong target here.**

```
function-level corpus   0.31B unique train tokens -> optimal model = 15.5M active
                        we ran exactly that (15M arm) and got 19/164
                        the 156M at 8.4x over-optimal got 31/164
file-level corpus       1.29B unique train tokens -> optimal model =  64M active
```

Loss-per-FLOP optimum is not the capability optimum. Do not use this rule to argue
for a smaller model.

**The docstring gap is healthy, not a failure.** Native 31 → stripped 24 = −22.6%.
Not zero (the model reads docstrings), not catastrophic (it does not need them).
Note the blind 24/164 still ties the old sighted 156M strict at 18/164.

**Whole-file training scored 0/164, twice.** Not a broken model: on 100 corpus
files cut after a docstring it wrote a body 17% of the time at 1 pass and 10% at 2
passes, against 88% for the same-size function-level model. That position is 0.04%
of file-level tokens vs 18% of function-level examples, so the signal is ~460×
thinner. A second pass over the same data moved val loss (1.3357 → 1.2847) and did
not move the behavior — epochs are ruled out, the objective is what has to change.
See the run 7/8 notes.

---

## 3. Datasets

### Strict (docstring required)

```
examples           752,420   (601,936 train / 75,242 val / 75,242 test)
train tokens/pass   93.4M
avg length         155.2 tokens
```

### Relaxed (docstring optional, `require_docstring: false`)

```
examples           4.18M total, 3.34M train   (measured from run step counts)
train tokens/pass  313M
avg length          93.5 tokens   (18% documented at 155.2, 82% bare at 80.2)
```

### File-level (`dataset_impl/python_file.py`, `max_length: 1024`)

Measured on 100k raw rows, plus a 30k-file random sample of the filtered corpus
spread across all 10 shards (an earlier single-shard sample read 229 tokens/file;
that was biased low by repo clustering — every shard now reads 285–292):

```
eligible files     72.6% of rows: 11.3% do not parse, 9.4% import outside the
                   stdlib+allowlist, 6.7% exceed 1024 tokens
                   (27% of files, but 60% of all characters)
length (<=1024)    mean 288, median 222, p75 392, p90 620, p99 958
fit <= 512         77% of eligible files     <- this config uses 1024
total per pass     5.59M files x 288 = 1.61B tokens
train per pass     1.29B tokens (80/10/10 split)
def/class bodies   only 15% of file characters
after a docstring  ~0.62M positions per pass = 0.04% of tokens
                   (0.11 documented top-level defs per file)
```

That last line is why run 7 scored zero. "Indent and write the body" is 18% of
function-level examples and one token in 2,500 at file level — about 460x less
signal on the exact move HumanEval tests.

The dropped long files are the interesting ones: 1025–2048 tokens is 6% of rows
but 42% more tokens and **63% more after-docstring positions** (0.83 documented
top-level defs per file vs 0.11 in the kept files). Raising the cap to 2048 moves
the density from 0.039% to 0.045%; dropping the import allowlist adds 14% tokens
and 24% positions (0.043%). Neither filter change moves the ratio that matters.

### The self-containment filter is the reason for file-level training

Measured on 20k real source rows:

```
candidate top-level functions   15,557
kept by extract_functions       10,368   (66.6%)
dropped: references a
  module-level name from the
  same file                     5,189   (33.4%)
dropped: multi-function etc.        0
```

A third of all functions are discarded because they call a helper or read a
constant defined elsewhere in the same file — i.e. the composed code.

### Cross-file imports are excluded today

```
repo metadata        present: repo_name and path columns
repo grouping        weak: median 1 file/repo, 74% single-file, p90 = 3, max 3,172
files importing
  non-stdlib roots   9.9% of parsed files   (rejected by the allowlist)
relative imports     0.4% of parsed files   (always rejected)
```

So the model sees zero project-local imports. Repo-level pretraining is possible
in principle but this corpus has almost no intact repos.

---

## 4. Measured performance

### Step-time model

Fitted from the 15M function-level run at batch 16 and 64, `max_length 512`:

```
step time = 0.0996 s  +  0.00087 s x batch
            ~fixed       per example
```

At batch 16, **~88% of a step is fixed overhead**, not model compute. The model is
10× smaller than the 156M but ran at 47.5M tokens/h vs its 53.4M tokens/h — a
proof that per-step overhead, not FLOPs, was the bottleneck.

Consequence: **batch size does not change total FLOPs at a fixed epoch count.** It
only changes the number of optimizer steps. Larger batch is nearly free when
updates are abundant.

### Throughput (measured)

```
156M active, batch 16, 512      53.4M train tokens/hour
13.1M active, batch 64, 512    ~176M train tokens/hour   (0.1224 s/step warm)
13.1M active, batch 16, 1024   ~165M train tokens/hour   (0.1010 s/step, 1.29B in 7.8 h)
13.1M active, batch 32, 1024   ~248M train tokens/hour   (0.134 s/step, 1.29B in 5.2 h)
228.2M active                  ~30.7M train tokens/hour  (extrapolated from 53.4M)
53.9M active                   ~130M train tokens/hour   (extrapolated, not yet run)
```

Batch 32 on `max_length 1024` is the fastest file-level config measured: nearly
double the batch-16 rate, because the fixed 0.0996 s overhead is paid half as often.

### GPU memory (measured, 24GB card)

```
15M, batch 16, 512        3,316 MB
15M, batch 64, 512       13,607 MB
15M, batch 16, 1024       6,543 MB
15M, batch 32, 1024      12,842 MB
156M, batch 16, 512      ~11,400 MB
```

---

## 5. Wall-clock and cost

```
function-level 15M  4 ep, batch 64, 512    ~7.8 h    (measured, 209,034 total steps)
function-level 156M 4 ep, batch 24, 512    ~22 h
file-level 15M      1 ep, batch 32, 1024   ~5.7 h    (measured, 139,767 steps; 5.2 h train)
file-level 15M      2 ep, batch 32, 1024   ~11.5 h   (measured, 279,534 steps; 11.2 h train)
270M relaxed        6 ep, batch 16, 512    ~58 h single / ~29 h dual  (estimated)
53.9M relaxed       4 ep, batch 64, 512    ~10 h     (estimated, not yet run)
```

Both file-level runs ran at batch 32, `max_length 1024`, ~12.8 GB. `epochs` was the
only difference between them. Run 7's config was never batch 16 — the batch 16
figures in section 4 are from the step-time sweep, not from these two runs.

---

## 6. Decisions and why

| decision | reason |
|---|---|
| `max_length: 512` for functions | 97.0% of documented and 98.7% of bare functions already fit; 1024 recovers 0.3% |
| `max_length: 1024` for files | 91.5% of eligible files fit; 512 would drop 23% and those are the multi-function files with helpers |
| `require_docstring: false` | 4.88× more functions; the only change that moved HumanEval materially |
| lr stays 6e-4 everywhere | below the critical batch size; sqrt-scaling adds a variable with no evidence |
| batch 16 for size ablations | matched recipe; larger batch only saves wall clock, never FLOPs |
| 4 epochs as the repetition boundary | Muennighoff: up to ~4 passes repetition is nearly free; beyond, it decays |
| do not use Chinchilla 20 tok/param | D is unique tokens; function corpus supports only 15.5M at that rule |
| `humaneval_truncate_at_top_level` | file-level models keep writing after a body; the cut reuses `_truncate_completion`. Not what cost run 7 its score: 82/164 outputs started a new top-level `def` and 70/164 were commented out, which fail either way |
| 2 epochs for the file re-run | the only free lever besides size; config diff against run 7 was exactly `epochs: 1` → `2`. It cost 11.5 h and bought nothing (17% → 10% body rate), which is what ruled epochs out |
| decontaminated | 0 harness repos, 52/164 name-matched, 1 file with ≥90% body overlap (a deliberately wrong solution) |

---

## 7. Known limitations

- **Sibling leakage.** Splits are by function, not by file, so sibling functions
  appear in train and val. Validation loss is optimistic and `best.pt` selection is
  noisy. HumanEval is external, so it is unaffected.
- **Quality dilution.** Removing the docstring filter also removed an implicit
  "somebody cared about this code" filter. 82% of the corpus is undocumented.
- **Estimates vs measurements.** Section 4 numbers marked extrapolated have not
  been measured on the actual card. Section 3 corpus totals are scaled from one
  shard (×~400), so absolute counts carry maybe ±10%; ratios do not.
- **The in-training compile benchmark is 2 samples.** `benchmark_num_samples: 2`
  means `bench_body=0.00` is a coin flip, not a signal. The 100-prefix test in
  section 1 is the measurement to trust.

---

## 8. Open questions

1. **Where is the size knee?** 13.1M → 19, 131M → 31. Next run: `h512 L10`
   (63.8M total / 53.9M active) on the relaxed corpus, ~10 h.
   `~28` = knee is low; `22–25` = knee near 50M; `18` = something else is wrong.
2. **Does whole-file training beat function training?** Still not answerable as a
   score: run 7 (1 pass, 0/164) and run 8 (2 passes, 0/164) against run 6 (function,
   4 passes, 19/164). But the epochs half of it is now answered — run 8 was run 7
   with only `epochs: 1` → `2`, and the body rate went 17% → 10% (n=100) while
   val_loss improved 1.3357 → 1.2847. Repetition does not teach this move, so a
   4th and 5th pass will not either; do not spend 18 h finding that out.
   What is left is the objective: mix the function-level task into file training, or
   use file-level as stage 1 and fine-tune on `signature -> body`. Only then is the
   4-pass, pass-matched comparison against run 6 worth running.
3. **Can 15M discover a rare pattern in file-level data?** Run 7 shows 15M can
   *represent* the docstring move (88% from the function-level model, n=100).
   Finding it from 0.04% of positions failed at 1 pass and at 2. A 131M file-level
   arm is now the only size-related test left, and it should follow an objective
   change, not precede it — a bigger model sees the same 0.04%.
4. **Does the 270M finally pay off?** On file-level data (1.29B/pass), 6 passes
   gives 34 tok/param, finally above Chinchilla. On function-level data it would
   need 14.7 passes, which is unreachable.
5. **Does inference-time context help?** Paste imported files' definitions into the
   prompt and see whether the model uses them. If it ignores injected context, no
   amount of file-level training makes retrieval work.

---

## 9. Interpretation notes

Gotchas that change how a number should be read. Each is a place where the obvious
reading of a result is the wrong one.

**"Data is the bottleneck" was too early.** Before measuring anything I wrote that
model size didn't matter because every arm scored the same. That was a
*measurement*, not an assumption — but it only covered 63.8M → 228.2M active on
one corpus. When the 15M arm scored 18 against the 156M's 31 on the same data, the
conclusion inverted: capacity binds, just below where I had looked. Lesson: a flat
sweep tells you the range you swept, not the shape of the curve.

**Chinchilla's 20 tok/param was a category error.** I first read it as "train
longer on the same data". It is not — `D` is *unique* tokens. Function corpus =
0.31B unique → optimal model = 15.5M active. The rule, applied honestly, says
train a model smaller than every arm already tested. It only became useful later
as a *ceiling* argument: the function corpus can never support a 270M model
(14.7 passes needed), while the file corpus can (6 passes → 27 tok/param).

**Batch size does not waste compute at a fixed epoch count.** I argued batch 24
"wasted" FLOPs. It doesn't: total FLOPs = 6 × active params × tokens, and tokens
per epoch is fixed by the dataset. Batch only re-chunks them. What it costs is
*optimizer updates*, and that only matters past the critical batch size, which
nobody here measured properly.

**I claimed the 15M run was data-loader-bound. It wasn't.** I extrapolated
"~3h total" from a dataloader floor. The actual first log line said 6.6 h/epoch.
The cause was per-step fixed overhead, which I had not modelled at all — see
section 4. The 10×-smaller model ran at *0.9×* the tokens/hour of the 156M. That
single line (`47.5M` vs `53.4M`) is the most useful diagnostic in this file.

**max_length for files.** I applied the function-level reasoning (512 covers
97%+, so 1024/2048 is pure cost) to files too. Wrong: files are 2–3× longer, and
512 drops 23% of them — exactly the multi-function files that motivated the
whole exercise. 1024 is correct there. Same knob, opposite answer, because the
unit of measurement changed.

**"Average file is 512 tokens" was wrong.** Measured properly (30k files sampled
across all shards): mean 288, median 222 — 1.8× below the guess. An early
single-shard sample read 229 and was itself low: files cluster by repo, so a
contiguous block is not a random sample. Sample across the whole corpus.

**The self-containment filter was invisible for weeks.** `extract_functions`
silently rejects any function that references a module-level name from its own
file — 33.4% of all candidates. It read as a correctness feature. It was also a
corpus-size decision nobody made on purpose. The most expensive mistakes are the
ones in a helper function you wrote early and never re-read.

**A score of exactly 0/164 made me check the harness before the model.** Right
reflex — the loader and script were both fine, two commands to prove it. But the
model was not broken either: it wrote plausible code, just commented out or as a
brand-new function. When a score is exactly zero, dump the raw generations before
forming an opinion about the weights. The interesting failure sat one line away
from the reported number and would never show up in the score itself.

**Token-matched is not progress-matched.** Runs 6 and 7 saw about the same tokens
(1.25B and 1.29B). Run 6 got 4 passes over its data, run 7 got 1, and the function
arm's validation loss was still falling at pass 4. The file-level config header said
"matched budget" as if that settled the comparison — it only settled the arithmetic.

**A falling validation loss did not mean it was learning the task.** Run 8 improved
val_loss from 1.3357 to 1.2847 (ppl 3.80 → 3.61) over run 7 and the behavior the
benchmark tests did not move at all — 17% → 10% body rate, both essentially zero
against the function-level model's 88%. Per-token loss averages over 1.3B positions,
so a signal occupying 0.04% of them can be completely absent from the number you
watch. When the task is rare in the data, the aggregate metric is the wrong
instrument: measure the move directly, or you will keep training against a curve
that has nothing to do with what you want.

**Two things I got right and would defend:** `max_length: 512` for functions
(measured, not guessed), and refusing to raise it to 1024/2048 when asked — the
recovery was 0.3% of examples.

---

## 10. Effect sizes, for quick reference

The comparisons that carry the argument. All n = 164.

```
4x unique data, same params      +13 HumanEval   (18 -> 31,  p=0.043)
3.5x params, same data            +0 HumanEval   (all landed in 18-20)
1.8x tokens (more epochs)         -2 HumanEval   (20 -> 18, within noise)
10x fewer params, 4x more data    +1 HumanEval   (18 vs 19, within noise)
whole files instead of functions -19 HumanEval   (19 -> 0, tokens matched 1.29B vs
                                                  1.25B — clean comparison)
1 more pass on file data          +0 HumanEval   (0 -> 0; val_loss -0.051, body
                                                  rate 17% -> 10%, n=100)
```

Read together: repetition and model size were both dead ends — including an extra
pass over file data, which moved val_loss and not the score — data volume was the
only lever that moved the number, and capacity only became visible once data
stopped being the bottleneck.

**Before quoting any of these:** run the paired McNemar over
`humaneval_samples*_eval_results.json`. The z-tests above treat the two runs as
independent samples, but both were scored on the same 164 problems, so the paired
test is both correct and more powerful. Everything here sits within ±2–3 problems
of noise, which is why the paired version matters.

---
