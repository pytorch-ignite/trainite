from pydantic import ConfigDict

# --- template-checking-only ---
import typing
from trainite.config.base import ProjectConfigBase

if typing.TYPE_CHECKING:
    from trainite.config.base import BaseModel, TrainerConfig

    ModelConfig = BaseModel
    DataConfig = BaseModel
    TrainerConfig = TrainerConfig
    PreprocessorConfig = BaseModel
# --- end-template-checking-only ---

# {{imports}}


class ProjectConfig(ProjectConfigBase):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    model: ModelConfig
    data: DataConfig
    trainer: TrainerConfig
    preprocessor: PreprocessorConfig | None = None
