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
| 1 | `gemma4_moe__vm` | 270.7M / 228.2M | strict, 5 ep | 467M | 20/164 (12.2%) | 12/164 (7.3%) | 60% |
| 2 | `gemma4_moe__vm` ×2 | 270.7M / 228.2M | strict, 2 ep | — | 13/164 | — | — |
| 3 | `config.ablation.yaml` | 155.8M / 131.0M | strict, 9 ep | 841M (best ≈607M) | 18/164 (11.0%) | 15/164 (9.1%) | 83% |
| 4 | `gemma4_moe__relaxed_h768` | 155.8M / 131.0M | relaxed, 4 ep | 1.20B | **31/164 (18.9%)** | **25/164 (15.2%)** | 81% |
| 5 | run 4, docstrings stripped | same weights | — | — | 24/164 (14.6%) | 19/164 (11.6%) | 79% |
| 6 | `config.relaxed_15m.yaml` | 15.4M / 13.1M | relaxed, 4 ep | 1.25B | 18/164 (11.0%) | 14/164 (8.5%) | 78% |
| 7 | `config.file_15m.yaml` | 15.4M / 13.1M | file-level, 1 ep | ~1.0B | *running* | *running* | — |

### Comparisons (two-proportion z-test, n = 164)

```
15M relaxed (18)  vs 156M relaxed (31)   same corpus, 10x params
  base  -7.9pp  z=-2.03  p=0.043
  plus  -6.7pp  z=-1.89  p=0.059

156M relaxed (31) vs 156M strict (18)    4x data, same params
  base  +7.9pp  z=+2.03  p=0.043
  plus  +6.1pp  z=+1.69  p=0.090

156M relaxed (31) vs 270M strict (20)    different corpus
  base  +6.7pp  z=+1.68  p=0.092
  plus  +7.9pp  z=+2.29  p=0.022
```

Wilson 95% intervals: 31/164 = 13.6–25.6%, 24/164 = 10.0–20.9%, 18/164 = 7.1–16.7%,
20/164 = 8.0–18.1%. ±2–3 solved is noise on 164 problems.

**Not yet done:** paired McNemar over `humaneval_samples*_eval_results.json`, which
is the right test because both runs are scored on the same 164 problems.

### Why each run exists

```
1  baseline. The "big model" everything else was measured against.
2  same model, fewer epochs — showed the val curve was still falling, so the
   5-epoch number was not a ceiling.
3  equal-compute ablation against run 1: 156M x 9 ep == 270M x 5 ep in FLOPs.
   Same score. This is where "size doesn't matter" came from — and where it
   stopped being true once the data changed.
4  the data experiment. Only the corpus changed. This is the headline.
5  the honesty check. Same weights, docstring removed from the prompt.
6  the capacity experiment. Only the size changed. This is what broke hypothesis 1.
7  the task experiment. Whole files instead of functions, matched token budget.
```

---

## 2. Headline conclusions

**Data volume was the binding constraint, and it was fixed.** 4× unique data bought
+13 problems; 3.5× parameters bought +0.

**Capacity binds between 13.1M and 131M active.** The earlier sweep (63.8M → 228.2M
active, all on strict data) found no difference. Now:

```
13.1M active  ->  18/164
131.0M active ->  31/164     same corpus, same recipe, only size differs
```

The knee is unmeasured. This is the most valuable open number.

**Data and capacity substitute, at roughly this exchange rate:**

```
156M strict  (18)  ==  15M relaxed (18)   10x less capacity, 4x more data, same score
+13 from 4x data (at 156M)  ==  +13 from 10x params (at relaxed)
```

**Chinchilla's 20 tok/param is the wrong target here.**

```
function-level corpus   0.31B unique train tokens -> optimal model = 15.5M active
                        we ran exactly that (15M arm) and got 18/164
                        the 156M at 8.4x over-optimal got 31/164
file-level corpus       1.02B unique train tokens -> optimal model =  51M active
```

Loss-per-FLOP optimum is not the capability optimum. Do not use this rule to argue
for a smaller model.

**The docstring gap is healthy, not a failure.** Native 31 → stripped 24 = −22.6%.
Not zero (the model reads docstrings), not catastrophic (it does not need them).
Note the blind 24/164 still ties the old sighted 156M strict at 18/164.

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

Measured on 40k real source rows of shard `train-00009-of-00027.parquet`:

```
eligible files     80% of rows (28% of the full corpus is not valid/allowlisted python)
length             mean 229, median 160, p75 325, p90 653, p99 2547, max 25710
fit <= 512         84.4%   mean 172
fit <= 1024        94.4%   mean 229     <- this config uses 1024
fit <= 2048        98.5%
total per pass     5.6M files x 229 = 1.28B tokens
train per pass     1.02B tokens
def/class bodies   only 15% of file characters
```

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
13.1M active, batch 16, 1024   ~114M train tokens/hour   (0.1010 s/step)
228.2M active                  ~30.7M train tokens/hour  (extrapolated from 53.4M)
53.9M active                   ~130M train tokens/hour   (extrapolated, not yet run)
```

### GPU memory (measured, 24GB card)

```
15M, batch 16, 512        3,316 MB
15M, batch 64, 512       13,607 MB
15M, batch 16, 1024       6,543 MB
156M, batch 16, 512      ~11,400 MB
```

---

## 5. Wall-clock and cost

```
function-level 15M  4 ep, batch 64, 512    ~7.8 h    (measured, 209,034 total steps)
function-level 156M 4 ep, batch 24, 512    ~22 h
file-level 15M      1 ep, batch 16, 1024   ~7.8 h    (measured so far, 279,534 steps)
file-level 15M      1 ep, batch 32, 1024   ~4.5 h    (estimated)
270M relaxed        6 ep, batch 16, 512    ~58 h single / ~29 h dual  (estimated)
53.9M relaxed       4 ep, batch 64, 512    ~10 h     (estimated, not yet run)
```

---

## 6. Decisions and why

| decision | reason |
|---|---|
| `max_length: 512` for functions | 97.0% of documented and 98.7% of bare functions already fit; 1024 recovers 0.3% |
| `max_length: 1024` for files | 94.4% fit; 512 would drop 15.6% and those are the multi-function files with helpers |
| `require_docstring: false` | 4.88× more functions; the only change that moved HumanEval materially |
| lr stays 6e-4 everywhere | below the critical batch size; sqrt-scaling adds a variable with no evidence |
| batch 16 for size ablations | matched recipe; larger batch only saves wall clock, never FLOPs |
| 4 epochs as the repetition boundary | Muennighoff: up to ~4 passes repetition is nearly free; beyond, it decays |
| do not use Chinchilla 20 tok/param | D is unique tokens; function corpus supports only 15.5M at that rule |
| `humaneval_truncate_at_top_level` | file-level models keep writing after a body; the cut reuses `_truncate_completion` |
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

---

## 8. Open questions

1. **Where is the size knee?** 13.1M → 18, 131M → 31. Next run: `h512 L10`
   (63.8M total / 53.9M active) on the relaxed corpus, ~10 h.
   `~28` = knee is low; `22–25` = knee near 50M; `18` = something else is wrong.
2. **Does whole-file training beat function training?** `config.file_15m.yaml` vs
   `config.relaxed_15m.yaml`, matched at ~1.0–1.25B tokens. Caveat: both arms are
   13M, which we now know is capacity-bound, so a tie does not settle the task
   question — it needs a 131M file-level arm.
3. **Does the 270M finally pay off?** On file-level data (1.02B/pass), 6 passes
   gives 27 tok/param, finally above Chinchilla. On function-level data it would
   need 14.7 passes, which is unreachable.
4. **Does inference-time context help?** Paste imported files' definitions into the
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
512 drops 15.6% of them — exactly the multi-function files that motivated the
whole exercise. 1024 is correct there. Same knob, opposite answer, because the
unit of measurement changed.

**"Average file is 512 tokens" was a 2× error.** Measured: mean 229, median 160.
The 2.2B-token figure it produced was 1.8× too high. Always measure the mean, not
the ceiling you configured.

**The self-containment filter was invisible for weeks.** `extract_functions`
silently rejects any function that references a module-level name from its own
file — 33.4% of all candidates. It read as a correctness feature. It was also a
corpus-size decision nobody made on purpose. The most expensive mistakes are the
ones in a helper function you wrote early and never re-read.

**Two things I got right and would defend:** `max_length: 512` for functions
(measured, not guessed), and refusing to raise it to 1024/2048 when asked — the
recovery was 0.3% of examples.

---

## 10. Effect sizes, for quick reference

The four comparisons that carry the argument. All n = 164.

```
4x unique data, same params      +13 HumanEval   (18 -> 31,  p=0.043)
3.5x params, same data            +0 HumanEval   (all landed in 18-20)
1.8x tokens (more epochs)         -2 HumanEval   (20 -> 18, within noise)
10x fewer params, 4x more data    +0 HumanEval   (18 vs 18, exact tie)
```

Read together: repetition and model size were both dead ends, data volume was the
only lever that moved the number, and capacity only became visible once data
stopped being the bottleneck.

**Before quoting any of these:** run the paired McNemar over
`humaneval_samples*_eval_results.json`. The z-tests above treat the two runs as
independent samples, but both were scored on the same 164 problems, so the paired
test is both correct and more powerful. Everything here sits within ±2–3 problems
of noise, which is why the paired version matters.

---
