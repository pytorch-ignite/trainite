# PreTrainer (Seq2Seq)

`PreTrainer` is the default training loop for this project, designed specifically for encoder-decoder models.

It is built on PyTorch-Ignite and already handles the things you usually want on day one:

- training and evaluation loops
- optimizer setup
- learning-rate warmup + decay
- checkpoints
- early stopping
- TensorBoard logging

## What it expects

`PreTrainer` works with a model that returns logits shaped like:

```python
(batch, tgt_seq_len, vocab_size)
```

and a batch that looks like:

```python
{
    "encoder_input_ids": Tensor,
    "decoder_input_ids": Tensor,
    "decoder_labels": Tensor,
}
```

It uses cross-entropy loss against `decoder_labels` and ignores label value `-100`.

## What happens during training

For each batch:

1. move batch tensors (`encoder_input_ids`, `decoder_input_ids`, `decoder_labels`) to device
2. run forward pass: `logits = model(encoder_input_ids=..., decoder_input_ids=...)`
3. compute cross-entropy loss against `decoder_labels`
4. backpropagate
5. clip gradients if `grad_clip_norm` is set
6. step optimizer

At the end of each epoch it also runs evaluation on:

- training data
- validation data, if present
- test data, if present

## Logging and checkpoints

`PreTrainer` saves a run directory like this:

```text
output/<run_name>/<timestamp>/
```

Inside it you get:

- `config.yaml`
- `output.log`
- `last_*` checkpoint
- `best_*` checkpoint if validation exists
- TensorBoard logs

## Config knobs

### `epochs`
How many full passes over the training data to run.

### `log_every_steps`
How often to print training updates.

### `grad_clip_norm`
If set, gradients are clipped before the optimizer step.

This can help when training becomes unstable.

## What to tweak first

If you are just getting started, the usual first changes are:

- `epochs`
- `log_every_steps`
- `grad_clip_norm`
- optimizer settings in `config.yaml`

## If you want to customize behavior

Edit `trainer.py` if you want to:

- change how loss is computed
- add gradient accumulation
- add new metrics
- change checkpoint rules
- change validation frequency
- change logging behavior

The main places to look are:

- `_train_step`
- `_eval_step`
- `_attach_metrics`
- `_attach_handlers`

## Minimal config example

```yaml
trainer:
  epochs: 10
  log_every_steps: 20
  grad_clip_norm: 1.0
```

## Good starting rule

If the run fails or looks unstable, check these first:

- input and label shapes match
- `vocab_size` is large enough for the dataset
- `encoder_max_seq_len` and `decoder_max_seq_len` are long enough
- learning rate is not too high
