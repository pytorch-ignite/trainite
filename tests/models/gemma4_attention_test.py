import torch

from trainite.models.gemma4_moe import Gemma4TextAttention


def test_gemma4_text_attention_forward():
    attention = Gemma4TextAttention(
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )
    hidden_states = torch.randn(2, 5, 32)

    output = attention(hidden_states)

    assert output.shape == hidden_states.shape
