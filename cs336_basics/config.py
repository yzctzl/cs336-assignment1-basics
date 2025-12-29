from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    train: str
    valid: str
    dtype: str = "uint16"


class ModelConfig(BaseModel):
    vocab_size: int = Field(gt=0)
    context_length: int = Field(gt=0)
    d_model: int = Field(multiple_of=2)
    num_layers: int = Field(ge=1)
    num_heads: int = Field(ge=1)
    d_ff: int
    rope_theta: float = 10000.0
    device: str = "cuda"


class OptimizerConfig(BaseModel):
    lr: float = Field(gt=0, le=1.0)
    betas: tuple[float, float] = (0.9, 0.99)
    eps: float = 1e-8
    weight_decay: float = 0.01


class CheckPointsConfig(BaseModel):
    enable: bool = False
    interval: int = Field(ge=1)
    to: str = "./dist/"


class TrainConfig(BaseModel):
    batch_size: int = Field(ge=1)
    steps: int = Field(gt=0)
    lr_max: float 
    lr_min: float
    t_w: int  # Warmup steps
    t_c: int  # Cosine decay steps
    grad_clip: float = 1.0
    accum_steps: int
    save: CheckPointsConfig


class Configures(BaseModel):
    seed: int = 42
    data: DataConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    train: TrainConfig
