# Char Tokenizer

`CharTokenizer` is a simple character-level tokenizer with a configurable vocabulary.

## Features

- Character-to-ID mapping from a charset preset or literal custom charset.
- Special tokens:
  - `<PAD>` (ID 0)
  - `<BOS>` (ID 1)
  - `<SEP>` (ID 2)
  - `<EOS>` (ID 3)
  - `<UNK>` (ID 4)
- Supports padding, truncation, and PyTorch tensor generation.

## Config knobs

### `_target_`

Must be set to `preprocessors.char_tokenizer.CharTokenizer` in the template config.

### `charset`

Accepts the same presets as the string-reverse dataset, including `@universal`, `@alpha`, `@digits`, and `@alphanumeric`. A literal string defines a custom vocabulary.

## Minimal config example

```yaml
preprocessor:
  _target_: preprocessors.char_tokenizer.CharTokenizer
  charset: "@alphanumeric"
```
