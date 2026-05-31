from pydantic import Field

from trainite.config.base import ComponentConfig


class DecoderModelConfig(ComponentConfig):
    target: str = Field(
        default="trainite.models.decoder.build_decoder_model", alias="_target_"
    )
    hidden_size: int = 64
    num_layers: int = 2
    num_heads: int = 2
    feedforward_dim: int = 128
    dropout: float = 0.1
    max_seq_len: int = 128



class EncoderDecoderModelConfig(ComponentConfig):
    target: str = Field(
        default="trainite.models.encoder_decoder.build_encoder_decoder_model", alias="_target_"
    )
    hidden_size: int = 64
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    num_heads: int = 2
    feedforward_dim: int = 128
    dropout: float = 0.1
    encoder_max_seq_len: int = 128
    decoder_max_seq_len: int = 128


