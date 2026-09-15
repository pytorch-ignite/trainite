import tempfile
from huggingface_hub import snapshot_download

import pytest
import torch


import json
from urllib.request import urlopen
from trainite.models.gemma4_moe import Gemma4TextModel
from transformers import Gemma4ForConditionalGeneration


@pytest.mark.load_weights
def test_tiny_hf_checkpoint_matches_transformers():
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint = snapshot_download("tiny-random/gemma-4-moe", local_dir=temp_dir)
        ours = Gemma4TextModel.from_hf_checkpoint(checkpoint).eval()
        reference = Gemma4ForConditionalGeneration.from_pretrained(checkpoint, local_files_only=True).eval()
        input_ids = torch.tensor([[2, 10, 11]])
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            actual = ours(input_ids, attention_mask).float()
            expected = reference(input_ids=input_ids, attention_mask=attention_mask).logits.float()

        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.004)
        assert torch.equal(actual.argmax(dim=-1), expected.argmax(dim=-1))


REVISION = "560dbcf0c2515abf83c1641b43e21bbcf178e2d7"
BASE_URL = f"https://huggingface.co/google/gemma-4-26B-A4B/resolve/{REVISION}"


def _download_json(filename: str) -> dict:
    with urlopen(f"{BASE_URL}/{filename}", timeout=30) as response:  # noqa: S310
        return json.load(response)


@pytest.mark.load_weights
def test_official_gemma4_26b_metadata_matches_loader():
    config = _download_json("config.json")["text_config"]
    weight_names = _download_json("model.safetensors.index.json")["weight_map"]

    assert config["hidden_size"] == 2816
    assert config["intermediate_size"] == 2112
    assert config["moe_intermediate_size"] == 704
    assert config["num_experts"] == 128
    assert config["top_k_experts"] == 8
    assert config["sliding_window"] == 1024
    assert config["layer_types"] == (["sliding_attention"] * 5 + ["full_attention"]) * 5

    local_rope = config["rope_parameters"]["sliding_attention"]
    global_rope = config["rope_parameters"]["full_attention"]
    assert local_rope["rope_theta"] == 10_000
    assert local_rope.get("partial_rotary_factor", 1.0) == 1.0
    assert global_rope["rope_theta"] == 1_000_000
    assert global_rope["partial_rotary_factor"] == 0.25

    required_suffixes = {
        "experts.down_proj",
        "experts.gate_up_proj",
        "input_layernorm.weight",
        "layer_scalar",
        "mlp.down_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "post_attention_layernorm.weight",
        "post_feedforward_layernorm.weight",
        "post_feedforward_layernorm_1.weight",
        "post_feedforward_layernorm_2.weight",
        "pre_feedforward_layernorm.weight",
        "pre_feedforward_layernorm_2.weight",
        "router.per_expert_scale",
        "router.proj.weight",
        "router.scale",
        "self_attn.k_norm.weight",
        "self_attn.k_proj.weight",
        "self_attn.o_proj.weight",
        "self_attn.q_norm.weight",
        "self_attn.q_proj.weight",
    }
    for layer_index, layer_type in enumerate(config["layer_types"]):
        prefix = f"model.language_model.layers.{layer_index}."
        expected = {prefix + suffix for suffix in required_suffixes}
        if layer_type == "sliding_attention":
            expected.add(prefix + "self_attn.v_proj.weight")
        assert expected <= weight_names.keys()

    assert "model.language_model.embed_tokens.weight" in weight_names
    assert "model.language_model.norm.weight" in weight_names
