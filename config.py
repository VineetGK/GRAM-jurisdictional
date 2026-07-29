"""
Configuration for GRAM-style Jurisdiction-Aware Legal Language Model
===================================================================

Install dependencies:
    pip install torch transformers datasets accelerate streamlit tiktoken tqdm

Run training:
    python train.py

Run evaluation:
    python evaluate.py

Run Streamlit app:
    streamlit run app.py
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

@dataclass
class Config:
    project_name: str = "gram-legal-llm"
    seed: int = 42
    device: str = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    
    data_dir: Path = Path("./data")
    tokenizer_dir: Path = Path("./tokenizer")
    checkpoint_dir: Path = Path("./checkpoints")
    log_dir: Path = Path("./logs")
    output_dir: Path = Path("./outputs")
    
    us_dataset_name: str = "free-law/Caselaw_Access_Project"
    eu_dataset_name: str = "joelniklaus/eurlex"
    general_dataset_name: str = "wikimedia/wikipedia"
    general_dataset_config: str = "20231101.en"
    
    max_samples_us: Optional[int] = 50000
    max_samples_eu: Optional[int] = 50000
    max_samples_general: Optional[int] = 10000
    
    tokenizer_vocab_size: int = 32000
    tokenizer_min_frequency: int = 2
    max_seq_len: int = 1024
    chunk_size: int = 512
    chunk_overlap: int = 50
    
    vocab_size: int = 32000
    max_seq_len: int = 1024
    n_layers: int = 12
    n_heads: int = 12
    d_model: int = 768
    d_ff: int = 3072
    dropout: float = 0.1
    module_mlp_ratio: float = 0.25
    
    enable_us_module: bool = True
    enable_eu_module: bool = True
    
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    max_steps: int = 5000
    warmup_steps: int = 500
    general_warmup_steps: int = 1000
    gram_steps: int = 4000
    
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    
    eval_interval: int = 500
    eval_steps: int = 100
    save_interval: int = 1000
    log_interval: int = 50
    
    eval_samples_us: int = 1000
    eval_samples_eu: int = 1000
    
    top_k: int = 50
    top_p: float = 0.9
    temperature: float = 0.8
    max_new_tokens: int = 256
    
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    
    use_amp: bool = True
    compile_model: bool = False
    
    def __post_init__(self):
        for dir_path in [self.data_dir, self.tokenizer_dir, self.checkpoint_dir, 
                         self.log_dir, self.output_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    @property
    def module_d_model(self) -> int:
        return int(self.d_model * self.module_mlp_ratio)
    
    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps
    
    def get_checkpoint_path(self, step: int) -> Path:
        return self.checkpoint_dir / f"checkpoint_step_{step}.pt"
    
    def get_latest_checkpoint(self) -> str:
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_step_*.pt"))
        if not checkpoints:
            return None
        return str(max(checkpoints, key=lambda x: int(x.stem.split("_")[-1])))

config = Config()