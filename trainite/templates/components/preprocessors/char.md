# Char Tokenizer

`CharTokenizer` is a simple character-level tokenizer with a hardcoded universal vocabulary.

## Features

- Character-to-ID mapping of printable ASCII characters plus space.
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

## Minimal config example

```yaml
preprocessor:
  _target_: preprocessors.char_tokenizer.CharTokenizer
```
