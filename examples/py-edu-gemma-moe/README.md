# Teaching a 15M Gemma 4 Model to Write Python

How much Python can a 15M-parameter decoder learn from scratch? This example trains Gemma 4 models
on Python functions and scores them on HumanEval+ mini, where the model gets a signature and docstring
and has to write the body.

The example trains two variants on the same corpus. They differ only in the feed-forward block: one
dense MLP, one mixture-of-experts with the same parameters active per token. That makes the MoE a
side question: does expert storage buy measurable Python ability at this scale, or not yet?

The base project was generated with `trainite==0.2.0`:

```bash
trainite init py-edu-gemma-moe --model gemma4-moe gemma4-dense --dataset python-edu --trainer decoder-trainer --run-name gemma4_moe__py_edu_15m
```

Customizations on top of the generated project:

1. **Preprocessor (`preprocessors/python_bpe.py`)**: a byte-level BPE fitted on the corpus replaces the
   GPT-2 tokenizer.
2. **Dataset (`dataset_impl/python_edu.py`)**: rows are parsed with `ast` and only self-contained
   top-level functions are kept, instead of training on whole files.
3. **Trainer (`trainer.py`)**: HumanEval+ mini evaluation via the official EvalPlus evaluator, run
   with and without the prompt's docstring.

## Task and data

[`Avelina/python-edu-cleaned`](https://huggingface.co/datasets/Avelina/python-edu-cleaned) is 6.05 GB
of Python source, 7,677,467 rows, Open Data Commons Attribution (ODC-BY). HumanEval hands the model a
signature and a docstring and asks for the body, so `load_function_dataset` turns each row into
prompt/completion pairs instead of training on whole files:

* Only top-level `def`/`async def`, no decorators and no inline suites.
* Only absolute imports, and only from the standard library plus a small allowlist of common data
  packages (`numpy`, `pandas`, `torch`, `sklearn`, ...).
* No reference to module-level globals the function does not define, so the body stands alone.
* The prompt is the file's imports plus the signature (and its docstring, when `require_docstring` is
  on); the completion is the body. `<eos>` is appended and prompt tokens are masked out of the loss.

The data splits 80/10/10 into train/val/test with `DataWithAutoSplit`. Sequences are capped at 512
tokens, with an 8192-token BPE vocabulary fitted once on the corpus.

Both configs set `require_docstring: false`, which roughly quadruples the usable corpus at the cost of
prompts that are only a signature. A single example:

```text
def add(a, b):
    return a + b
```

## The two models

| | `configs/dense.yaml` | `configs/moe.yaml` |
|---|---|---|
| model | `Gemma4DenseModel` | `Gemma4MoEModel` |
| feed-forward | `dense_intermediate_size: 1536` | `dense_intermediate_size: 768`, `num_experts: 3`, `expert_dim: 384`, `top_k: 2` |
| loss | `CausalLMCrossEntropyLoss` | `GemmaMoECausalLMLoss`, `aux_loss_coef: 0.01` |
| parameters, total | 13,051,400 | 15,425,056 |
| parameters per token | 13,051,400 | 13,065,760 |
| training time, 4 epochs | 7h57m | 12h27m |
| seconds per step | 0.1380 | 0.2144 |

Both arms share the rest of the config: 8 layers, hidden size 256, 4 attention heads with 2 KV heads,
head dimension 64, the `sssg` sliding/global pattern with a 512-token window, batch size 64, AdamW at
6e-4, bf16, 4 epochs, one RTX 3090 each.

## Metrics

* `loss`: teacher-forced cross-entropy on completion tokens.
* `token_accuracy`: token-level accuracy on completion tokens.
* HumanEval+ mini (`humaneval/...`): 164 problems, greedy decoding, scored by EvalPlus' official
  evaluator. Each checkpoint solves the set twice, once with the docstring in the prompt and once with
  it stripped, which separates code synthesis from copying the description.

## Results

| | dense | MoE |
|---|---|---|
| validation loss | 1.0639 | 1.0637 |
| token accuracy | 0.7494 | 0.7495 |
| HumanEval+ base, docstring | 15/164 | 16/164 |
| HumanEval+ base, stripped | 19/164 | 18/164 |
| HumanEval+ plus, docstring | 12/164 | 13/164 |
| HumanEval+ plus, stripped | 16/164 | 15/164 |

The pretraining run does not separate the two models. The extra 2.4M expert parameters moved the
validation loss by 0.0002 and HumanEval by one task, while costing 1.6x the training time. The experts
are storage, not arithmetic: 768 shared + 2 of 3 x 384 = 1536 per token, exactly the dense FFN width,
so both arms do the same work per token.

That is a statement about this budget, not about MoE in general. Both arms stopped after four
pretraining epochs on next-token prediction, and neither was post-trained. Nothing here rules out the
experts becoming useful under more training or a later code post-training stage; it only says the
storage on its own does not pay off yet.

### Noise floor

The counts come from 164 problems, so they are a sample, not a measurement. At the observed 9-12%
pass rate the binomial standard error is `sqrt(p * (1 - p) / 164)`, or 2.3-2.5 points (3.8-4.1
tasks). A one-task gap is well inside that.

Pairing the models on the same problems does not change the answer. With a one-task margin the
discordant counts differ by one, so an exact McNemar test returns `p = 1` for every possible
per-problem pairing. The docstring gaps, 15 vs 19 and 16 vs 18, are not significant either: even the
most favorable pairing gives `p = 0.125` and `p = 0.5`.

That covers sampling noise only. Each arm is a single seed, so how much HumanEval moves between
training runs, usually the larger term, is not measured here. The per-task outcomes a paired test
needs are written next to each run as `humaneval_samples*_eval_results.json`.

## Run the example

From this directory:

```bash
uv sync
uv run python main.py configs/dense.yaml
uv run python main.py configs/moe.yaml
```

The configs log to ClearML, so watch the run in the ClearML web UI. For local TensorBoard instead, set
`logger: tensorboard` and run:

```bash
uv run tensorboard --logdir outputs
```

Checkpoints and logs are written below `outputs/<run_name>/<timestamp>/`.

### Cloud

`sky.yaml` runs the same command on a rented GPU:

```bash
sky check
sky launch sky.yaml
```

Set `envs.CONFIG` to pick the arm, and edit the resource block to change the accelerator, price cap, or
provider (`cloud: aws`, `cloud: vast`, ...). Without a `cloud` entry SkyPilot picks any provider that
satisfies the request.

> [!NOTE]
> The numbers above were produced on Vast.ai with a small private `vastai` launcher, not
> `sky launch`: SkyPilot's Vast backend could not list or destroy instances and attached the wrong SSH
> keys at the time. `sky.yaml` is the portable path and works on any provider; the custom launcher is
> not part of this example.

## Walkthrough notebook

`walkthrough.ipynb` walks through the data, the collator, both architectures, and inference. It runs
on a 5,000-example slice of the corpus and loads the trained weights from
[`Blankyy/py-edu-gemma-15m`](https://huggingface.co/Blankyy/py-edu-gemma-15m), so it needs no ClearML
credentials. The weights and tokenizer (~114 MB) are cached by `huggingface_hub` on first run.

Open it from this directory with the example's `.venv` kernel.

## Files

* `configs/dense.yaml`, `configs/moe.yaml` — the two arms.
* `main.py` — single-run entry point.
* `trainer.py` — training loop, metrics, checkpoints, HumanEval+ evaluation.
* `config.py`, `utils.py` — config schema and wiring.
* `models/gemma4_dense.py`, `models/gemma4_moe.py` — the two architectures.
* `dataset_impl/python_edu.py` — function extraction and tokenization.
* `preprocessors/python_bpe.py` — corpus-trained BPE tokenizer.
* `walkthrough.ipynb` — end-to-end walkthrough with the published weights.
* `sky.yaml` — SkyPilot cloud launch.
