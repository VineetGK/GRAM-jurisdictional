import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    # Project paths
    project_root: Path = Path(__file__).parent
    data_dir: Path = project_root / "data"
    tokenizer_dir: Path = project_root / "tokenizer"
    checkpoints_dir: Path = project_root / "checkpoints"
    logs_dir: Path = project_root / "logs"

    # HuggingFace token for gated datasets
    hf_token: Optional[str] = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

    # Model hyperparameters (small GPT-style, ~100M params)
    vocab_size: int = 32000
    max_seq_len: int = 512
    n_layers: int = 12
    n_heads: int = 12
    n_embd: int = 768
    n_inner: int = 3072
    dropout: float = 0.1
    attn_dropout: float = 0.1
    resid_dropout: float = 0.1
    embd_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    n_positions: int = 512
    n_ctx: int = 512

    # GRAM module config
    enable_us_module: bool = True
    enable_eu_module: bool = True
    module_mlp_ratio: float = 0.25  # module MLP size as fraction of core MLP

    # Tokenizer config
    tokenizer_type: str = "bpe"  # "bpe" or "wordpiece"
    vocab_size: int = 32000
    min_frequency: int = 2
    special_tokens: list = field(default_factory=lambda: ["<pad>", "<unk>", "<bos>", "<eos>", "<us>", "<eu>", "<gen>"])

    # Data config
    max_seq_len: int = 512
    chunk_size: int = 512
    chunk_overlap: int = 50
    train_split: float = 0.9
    val_split: float = 0.1
    test_split: float = 0.0

    # Dataset configs (placeholder paths - replace with actual data)
    us_data_path: Optional[Path] = None  # e.g., data/us_caselaw/
    eu_data_path: Optional[Path] = None  # e.g., data/eurlex/
    general_data_path: Optional[Path] = None  # e.g., data/wikipedia/

    # Sample sizes for quick prototyping
    us_sample_size: int = 10000
    eu_sample_size: int = 10000
    general_sample_size: int = 5000

    # Training config
    batch_size: int = 16
    micro_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_steps: int = 10000
    warmup_steps: int = 1000
    eval_interval: int = 500
    save_interval: int = 1000
    log_interval: int = 100

    # Training phases
    phase1_general_warmup_steps: int = 2000  # Phase 1: general warm-up
    phase2_gram_steps: int = 8000  # Phase 2: GRAM phase

    # Optimizer
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    max_grad_norm: float = 1.0

    # Learning rate schedule
    lr_schedule: str = "cosine"  # "cosine", "linear", "constant"
    min_lr: float = 3e-5

    # GRAM training config
    grad_routing: bool = True
    freeze_core_during_gram: bool = True
    freeze_other_module_during_gram: bool = True
    general_updates_core_only: bool = False  # if True, general updates only core

    # Evaluation config
    eval_batch_size: int = 32
    eval_steps: int = 100
    eval_max_tokens: int = 512

    # Evaluation configs for ablation
    eval_configs: list = field(default_factory=lambda: [
        {"name": "Full", "enable_us_module": True, "enable_eu_module": True},
        {"name": "US-only", "enable_us_module": True, "enable_eu_module": False},
        {"name": "EU-only", "enable_us_module": False, "enable_eu_module": True},
    ])

    # Evaluation datasets
    eval_us_sample_size: int = 1000
    eval_eu_sample_size: int = 1000

    # Generation config (for eval and app)
    gen_max_new_tokens: int = 256
    gen_temperature: float = 0.8
    gen_top_k: int = 50
    gen_top_p: float = 0.95
    gen_repetition_penalty: float = 1.1

    # Streamlit app config
    app_title: str = "GRAM for Law - Legal Assistant"
    app_mode_options: list = field(default_factory=lambda: [
        "Full (US + EU)",
        "US-only",
        "EU-only",
    ])
    app_mode_to_config: dict = field(default_factory=lambda: {
        "Full (US + EU)": {"enable_us_module": True, "enable_eu_module": True},
        "US-only": {"enable_us_module": True, "enable_eu_module": False},
        "EU-only": {"enable_us_module": False, "enable_eu_module": True},
    })

    # Device and distributed
    device: str = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    dtype: str = "bfloat16" if __import__("torch").cuda.is_available() and __import__("torch").cuda.is_bf16_supported() else "float16"
    compile_model: bool = False  # torch.compile
    seed: int = 42

    # Logging
    wandb_project: Optional[str] = "gram-law"
    wandb_entity: Optional[str] = None
    wandb_log: bool = False

    def __post_init__(self):
        # Create directories
        for d in [self.data_dir, self.tokenizer_dir, self.checkpoints_dir, self.logs_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Set default data paths if not provided
        if self.us_data_path is None:
            self.us_data_path = self.data_dir / "us_caselaw"
        if self.eu_data_path is None:
            self.eu_data_path = self.data_dir / "eurlex"
        if self.general_data_path is None:
            self.general_data_path = self.data_dir / "wikipedia"

        # Derived
        self.gradient_accumulation_steps = max(1, self.batch_size // self.micro_batch_size)

    def get_dtype(self):
        import torch
        if self.dtype == "bfloat16":
            return torch.bfloat16
        elif self.dtype == "float16":
            return torch.float16
        return torch.float32


config = Config()