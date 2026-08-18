# GPT-2 Tokenizer

This project uses the fast GPT-2 tokenizer from Hugging Face Transformers. Its byte-level BPE vocabulary handles
general text more efficiently than the character tokenizer.

GPT-2 has no dedicated padding token, so `load_gpt2_tokenizer` reuses its end-of-text token for padding. Trainite's
attention masks still distinguish padding from real end-of-text tokens.

## Minimal config

```yaml
preprocessor:
  _target_: preprocessors.gpt2_tokenizer.load_gpt2_tokenizer
```

Edit `preprocessors/gpt2_tokenizer.py` if you need another pretrained tokenizer. Keep in mind that a larger vocabulary
also makes Trainite's embedding and output layers larger.
