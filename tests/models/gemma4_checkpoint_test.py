import json
from pathlib import Path
import tempfile
from urllib.request import urlopen

import pytest
from safetensors.torch import load_file
import torch
from huggingface_hub import snapshot_download
from transformers import Gemma4ForConditionalGeneration

from trainite.models.gemma4_dense import Gemma4DenseModel
from trainite.models.gemma4_moe import Gemma4MoEModel


def _load_hf_gemma4_dense_model(checkpoint_dir: str | Path) -> Gemma4DenseModel:
    """Load an unsharded Hugging Face Gemma 4 Dense text checkpoint."""

    checkpoint_dir = Path(checkpoint_dir)
    with (checkpoint_dir / "config.json").open(encoding="utf-8") as file:
        raw_config = json.load(file)
        config = raw_config.get("text_config", raw_config)

    rope = config["rope_parameters"]
    local_rope = rope["sliding_attention"]
    global_rope = rope["full_attention"]
    layer_types = tuple(
        "sliding" if layer_type == "sliding_attention" else "global" for layer_type in config["layer_types"]
    )
    tie_word_embeddings = config.get("tie_word_embeddings", True)
    model = Gemma4DenseModel(
        vocab_size=config["vocab_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_hidden_layers"],
        num_attention_heads=config["num_attention_heads"],
        num_key_value_heads=config["num_key_value_heads"],
        dense_intermediate_size=config["intermediate_size"],
        head_dim=config["head_dim"],
        rope_theta=local_rope["rope_theta"],
        rope_scaling_factor=local_rope.get("factor", 1.0),
        rotary_fraction=local_rope.get("partial_rotary_factor", 1.0),
        attention_dropout=config.get("attention_dropout", 0.0),
        pad_token_id=config.get("pad_token_id"),
        layer_types=layer_types,
        sliding_window=config["sliding_window"],
        global_num_key_value_heads=config["num_global_key_value_heads"],
        global_head_dim=config["global_head_dim"],
        global_rope_theta=global_rope["rope_theta"],
        global_rope_scaling_factor=global_rope.get("factor", 1.0),
        global_rotary_fraction=global_rope.get("partial_rotary_factor", 1.0),
        global_key_equals_value=config.get("attention_k_eq_v", False),
        tie_word_embeddings=tie_word_embeddings,
        final_logit_softcap=config.get("final_logit_softcapping"),
    )

    source = load_file(checkpoint_dir / "model.safetensors", device="cpu")
    converted = {
        "token_embedding.weight": source["model.language_model.embed_tokens.weight"],
        "final_norm.weight": source["model.language_model.norm.weight"],
    }
    if not tie_word_embeddings and "lm_head.weight" in source:
        converted["lm_head.weight"] = source["lm_head.weight"]

    layer_key_map = {
        "layer_scalar": "layer_scale",
        "self_attn.q_proj.weight": "attention.q_proj.weight",
        "self_attn.k_proj.weight": "attention.k_proj.weight",
        "self_attn.v_proj.weight": "attention.v_proj.weight",
        "self_attn.o_proj.weight": "attention.o_proj.weight",
        "self_attn.q_norm.weight": "attention.q_norm.weight",
        "self_attn.k_norm.weight": "attention.k_norm.weight",
        "input_layernorm.weight": "pre_attention_norm.weight",
        "post_attention_layernorm.weight": "post_attention_norm.weight",
        "pre_feedforward_layernorm.weight": "pre_dense_norm.weight",
        "mlp.gate_proj.weight": "dense_mlp.gate_proj.weight",
        "mlp.up_proj.weight": "dense_mlp.up_proj.weight",
        "mlp.down_proj.weight": "dense_mlp.down_proj.weight",
        "post_feedforward_layernorm.weight": "post_dense_norm.weight",
    }
    for layer_index in range(config["num_hidden_layers"]):
        source_prefix = f"model.language_model.layers.{layer_index}."
        target_prefix = f"layers.{layer_index}."
        for source_suffix, target_suffix in layer_key_map.items():
            source_key = source_prefix + source_suffix
            if source_key in source:
                converted[target_prefix + target_suffix] = source[source_key]

    model.to(dtype=converted["token_embedding.weight"].dtype)
    model.load_state_dict(converted, strict=True)
    return model


def _load_hf_gemma4_text_model(checkpoint_dir: str | Path) -> Gemma4MoEModel:
    """Load an unsharded Hugging Face Gemma 4 MoE text checkpoint."""

    checkpoint_dir = Path(checkpoint_dir)
    with (checkpoint_dir / "config.json").open(encoding="utf-8") as file:
        config = json.load(file)["text_config"]

    rope = config["rope_parameters"]
    local_rope = rope["sliding_attention"]
    global_rope = rope["full_attention"]
    layer_types = tuple(
        "sliding" if layer_type == "sliding_attention" else "global" for layer_type in config["layer_types"]
    )
    model = Gemma4MoEModel(
        vocab_size=config["vocab_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_hidden_layers"],
        num_attention_heads=config["num_attention_heads"],
        num_key_value_heads=config["num_key_value_heads"],
        dense_intermediate_size=config["intermediate_size"],
        expert_dim=config["moe_intermediate_size"],
        num_experts=config["num_experts"],
        top_k=config["top_k_experts"],
        head_dim=config["head_dim"],
        rope_theta=local_rope["rope_theta"],
        rope_scaling_factor=local_rope.get("factor", 1.0),
        rotary_fraction=local_rope.get("partial_rotary_factor", 1.0),
        attention_dropout=config.get("attention_dropout", 0.0),
        pad_token_id=config.get("pad_token_id"),
        layer_types=layer_types,
        sliding_window=config["sliding_window"],
        global_num_key_value_heads=config["num_global_key_value_heads"],
        global_head_dim=config["global_head_dim"],
        global_rope_theta=global_rope["rope_theta"],
        global_rope_scaling_factor=global_rope.get("factor", 1.0),
        global_rotary_fraction=global_rope.get("partial_rotary_factor", 1.0),
        global_key_equals_value=config.get("attention_k_eq_v", False),
        final_logit_softcap=config.get("final_logit_softcapping"),
    )

    source = load_file(checkpoint_dir / "model.safetensors", device="cpu")
    converted = {
        "token_embedding.weight": source["model.language_model.embed_tokens.weight"],
        "final_norm.weight": source["model.language_model.norm.weight"],
    }
    layer_key_map = {
        "layer_scalar": "layer_scale",
        "self_attn.q_proj.weight": "attention.q_proj.weight",
        "self_attn.k_proj.weight": "attention.k_proj.weight",
        "self_attn.v_proj.weight": "attention.v_proj.weight",
        "self_attn.o_proj.weight": "attention.o_proj.weight",
        "self_attn.q_norm.weight": "attention.q_norm.weight",
        "self_attn.k_norm.weight": "attention.k_norm.weight",
        "input_layernorm.weight": "pre_attention_norm.weight",
        "post_attention_layernorm.weight": "post_attention_norm.weight",
        "pre_feedforward_layernorm.weight": "pre_dense_norm.weight",
        "mlp.gate_proj.weight": "dense_mlp.gate_proj.weight",
        "mlp.up_proj.weight": "dense_mlp.up_proj.weight",
        "mlp.down_proj.weight": "dense_mlp.down_proj.weight",
        "post_feedforward_layernorm_1.weight": "post_dense_norm.weight",
        "pre_feedforward_layernorm_2.weight": "pre_moe_norm.weight",
        "router.scale": "moe.router_scale",
        "router.per_expert_scale": "moe.per_expert_scale",
        "router.proj.weight": "moe.router.weight",
        "experts.gate_up_proj": "moe.gate_up_proj",
        "experts.down_proj": "moe.down_proj",
        "post_feedforward_layernorm_2.weight": "post_moe_norm.weight",
        "post_feedforward_layernorm.weight": "post_feedforward_norm.weight",
    }
    for layer_index in range(config["num_hidden_layers"]):
        source_prefix = f"model.language_model.layers.{layer_index}."
        target_prefix = f"layers.{layer_index}."
        for source_suffix, target_suffix in layer_key_map.items():
            source_key = source_prefix + source_suffix
            if source_key in source:
                converted[target_prefix + target_suffix] = source[source_key]

    model.to(dtype=converted["token_embedding.weight"].dtype)
    model.load_state_dict(converted, strict=True)
    return model


@pytest.mark.model_parity
def test_tiny_hf_checkpoint_matches_transformers():
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint = snapshot_download("tiny-random/gemma-4-moe", local_dir=temp_dir)
        ours = _load_hf_gemma4_text_model(checkpoint).float().eval()
        reference = Gemma4ForConditionalGeneration.from_pretrained(checkpoint, local_files_only=True).float().eval()
        input_ids = torch.tensor([[2, 10, 11]])
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            actual = ours(input_ids, attention_mask).float()
            expected = reference(input_ids=input_ids, attention_mask=attention_mask).logits.float()

        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
        assert torch.equal(actual.argmax(dim=-1), expected.argmax(dim=-1))


@pytest.mark.model_parity
def test_tiny_hf_dense_checkpoint_matches_transformers():
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint = snapshot_download("tiny-random/gemma-4-dense", local_dir=temp_dir)
        ours = _load_hf_gemma4_dense_model(checkpoint).float().eval()
        reference = Gemma4ForConditionalGeneration.from_pretrained(checkpoint, local_files_only=True).float().eval()
        input_ids = torch.tensor([[2, 10, 11]])
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            actual = ours(input_ids, attention_mask).float()
            expected = reference(input_ids=input_ids, attention_mask=attention_mask).logits.float()

        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
        assert torch.equal(actual.argmax(dim=-1), expected.argmax(dim=-1))


REVISION = "560dbcf0c2515abf83c1641b43e21bbcf178e2d7"
BASE_URL = f"https://huggingface.co/google/gemma-4-26B-A4B/resolve/{REVISION}"


def _download_json(filename: str) -> dict:
    with urlopen(f"{BASE_URL}/{filename}", timeout=30) as response:  # noqa: S310
        return json.load(response)


@pytest.mark.model_parity
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
