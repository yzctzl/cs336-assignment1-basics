from typing import Any

from pydantic import BaseModel, Field, model_validator


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
    d_ff: int | None = None
    rope_theta: float | None = 10000.0
    device: str = "cuda"

    @model_validator(mode="after")
    def validate_d_ff(self) -> "ModelConfig":
        if self.d_ff is None:
            # Default scaling: 8/3 * d_model, aligned to 64/128
            aligned = 128
            self.d_ff = int((2 * (4 * self.d_model) / 3 + aligned) // aligned) * aligned
        return self


class OptimizerConfig(BaseModel):
    type: str = "muon"  # or adamw
    muon_lr: float | None = 2e-2
    muon_momentum: float | None = 0.95
    lr: float = Field(gt=0, le=1.0)  # must same as TrainConfig.lr_max!
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
    lr_scheduler: str = "wsd"  # or cosine
    lr_max: float
    lr_min: float
    t_w: int  # Warmup steps
    t_c: int  # Cosine decay steps
    grad_clip: float = 1.0
    accum_steps: int
    save: CheckPointsConfig


class TokenizerConfig(BaseModel):
    vocab_path: str
    merges_path: str
    special_tokens: list[str]


class InferConfig(BaseModel):
    kv_cache: bool
    checkpoint: str


class Configures(BaseModel):
    seed: int = 32
    data: DataConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    train: TrainConfig
    tokenizer: TokenizerConfig
    infer: InferConfig

    @model_validator(mode="after")
    def validate_lr_consistency(self) -> "Configures":
        if self.optimizer.lr != self.train.lr_max:
            print(
                f"optimizer.lr ({self.optimizer.lr}) must be equal to train.lr_max ({self.train.lr_max})"
            )
            self.optimizer.lr = self.train.lr_max
        return self

    @model_validator(mode="after")
    def validate_wsd_tc(self) -> "Configures":
        if self.train.lr_scheduler == "wsd":
            decay_ratio = 0.1
            expected_t_c = int(self.train.steps * (1.0 - decay_ratio))
            if self.train.t_c != expected_t_c:
                raise ValueError(
                    f"For 'wsd' scheduler, train.t_c must be {1.0 - decay_ratio} * train.steps "
                    f"(expected {expected_t_c}, got {self.train.t_c})"
                )
        return self



def update_cfg_w_sweep(base_config: Configures, sweep_config: dict[str, Any]) -> Configures:
    config: dict = base_config.model_dump()

    for key, value in sweep_config.items():
        if key.startswith("_"):
            continue

        parts = key.split(".")
        target = config
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value

    sweep_keys = sweep_config.keys()
    if "model.d_model" in sweep_keys and "model.d_ff" not in sweep_keys:
        if "model" in config:
            config["model"]["d_ff"] = None

    return Configures.model_validate(config)
