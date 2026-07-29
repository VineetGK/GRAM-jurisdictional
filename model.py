"""
GRAM Legal LLM Model
=====================

GRAM (Gradient Routed Auxiliary Modules) architecture:
- Shared core Transformer backbone (domain-general legal knowledge)
- Jurisdiction-specific auxiliary MLP modules (US, EU) - extra neurons per layer
- Gradient routing during training: core + active jurisdiction module only
- Inference: select core + relevant jurisdiction module(s)

Architecture:
```
Input -> Embedding -> Core Transformer Blocks -> [Core + Jurisdiction Modules] -> LM Head -> Logits
                                                |
                                                +-- US Module (extra MLP neurons)
                                                +-- EU Module (extra MLP neurons)
```

Training (Gradient Routing):
- US batch: freeze core + EU module, only US module gets gradients
- EU batch: freeze core + US module, only EU module gets gradients  
- General batch: unfreeze core + both modules
"""

import math
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm

from config import config


Jurisdiction = Literal["US", "EU", "general", "core"]


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    max_seq_len: int = 1024
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    d_ff: int = 3072
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    
    module_mlp_ratio: float = 0.25
    jurisdictions: List[Jurisdiction] = None
    
    def __post_init__(self):
        if self.jurisdictions is None:
            self.jurisdictions = ["US", "EU", "general"]
        assert self.d_model % self.n_heads == 0
    
    @property
    def module_d_model(self) -> int:
        return int(self.d_model * self.module_mlp_ratio)


def get_model_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=config.tokenizer_vocab_size,
        max_seq_len=config.max_seq_len,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        dropout=config.dropout,
        layer_norm_eps=config.layer_norm_eps,
        module_mlp_ratio=config.module_mlp_ratio,
        jurisdictions=config.jurisdictions,
    )


class MultiHeadAttention(nn.Module):
    """Standard multi-head attention."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads
        
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        
        self.dropout = nn.Dropout(config.dropout)
        
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len)).view(
                1, 1, config.max_seq_len, config.max_seq_len
            ),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        
        causal_mask = self.causal_mask[:, :, :seq_len, :seq_len]
        attn_scores = attn_scores.masked_fill(causal_mask == 0, float("-inf"))
        
        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(
                attention_mask.view(batch_size, 1, 1, seq_len) == 0, float("-inf")
            )
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        output = self.out_proj(attn_output)
        return output


class CoreMLP(nn.Module):
    """Core MLP (shared across all jurisdictions)."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.d_model, config.d_ff)
        self.fc2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class JurisdictionMLP(nn.Module):
    """Jurisdiction-specific auxiliary MLP (extra neurons)."""
    
    def __init__(self, config: ModelConfig, jurisdiction: Jurisdiction):
        super().__init__()
        self.jurisdiction = jurisdiction
        self.config = config
        self.module_d_model = config.module_d_model
        
        self.fc1 = nn.Linear(config.d_model, self.module_d_model)
        self.fc2 = nn.Linear(self.module_d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = nn.GELU()
        
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    """Single Transformer block with core MLP + optional jurisdiction MLPs."""
    
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        self.ln1 = LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.attention = MultiHeadAttention(config)
        self.ln2 = LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.core_mlp = CoreMLP(config)
        self.dropout = nn.Dropout(config.dropout)
        
        self.us_module = JurisdictionMLP(config, "US")
        self.eu_module = JurisdictionMLP(config, "EU")
        self.general_module = JurisdictionMLP(config, "general")
        
        self.enable_us_module: bool = True
        self.enable_eu_module: bool = True
        self.enable_general_module: bool = True
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attn_out = self.attention(self.ln1(x), attention_mask)
        x = x + self.dropout(attn_out)
        
        core_out = self.core_mlp(self.ln2(x))
        
        module_out = torch.zeros_like(core_out)
        if self.enable_us_module:
            module_out = module_out + self.us_module(x)
        if self.enable_eu_module:
            module_out = module_out + self.eu_module(x)
        if self.enable_general_module:
            module_out = module_out + self.general_module(x)
        
        x = x + self.dropout(core_out + module_out)
        
        return x


class GRAMModel(nn.Module):
    """
    GRAM (Gradient Routed Auxiliary Modules) Legal Language Model.
    
    Architecture:
    - Shared core Transformer backbone
    - Jurisdiction-specific auxiliary MLPs per layer (extra neurons)
    - Gradient routing during training (core + active jurisdiction only)
    - Inference: select core + relevant jurisdiction module(s)
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(config, i) for i in range(config.n_layers)
        ])
        
        self.ln_f = LayerNorm(config.d_model, eps=config.layer_norm_eps)
        
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        
        self.active_jurisdiction: Optional[Jurisdiction] = "core"
        self.enable_us_module: bool = True
        self.enable_eu_module: bool = True
        self.enable_general_module: bool = True
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def set_jurisdiction(self, jurisdiction: Jurisdiction):
        """Set active jurisdiction for inference."""
        assert jurisdiction in ["core"] + self.config.jurisdictions
        self.active_jurisdiction = jurisdiction
        
        for block in self.blocks:
            if jurisdiction == "core":
                block.enable_us_module = False
                block.enable_eu_module = False
                block.enable_general_module = False
            elif jurisdiction == "US":
                block.enable_us_module = True
                block.enable_eu_module = False
                block.enable_general_module = False
            elif jurisdiction == "EU":
                block.enable_us_module = False
                block.enable_eu_module = True
                block.enable_general_module = False
            elif jurisdiction == "general":
                block.enable_us_module = False
                block.enable_eu_module = False
                block.enable_general_module = True
    
    def set_module_config(self, enable_us: bool, enable_eu: bool, enable_general: bool = False):
        """Set module configuration for inference (Full/US-only/EU-only)."""
        self.enable_us_module = enable_us
        self.enable_eu_module = enable_eu
        self.enable_general_module = enable_general
        
        for block in self.blocks:
            block.enable_us_module = enable_us
            block.enable_eu_module = enable_eu
            block.enable_general_module = enable_general
    
    def route_gradients(self, jurisdiction: Jurisdiction):
        """
        GRAM gradient routing: freeze core + other jurisdiction modules,
        only allow gradients for active jurisdiction module.
        """
        for name, param in self.named_parameters():
            if "us_module" in name:
                param.requires_grad = (jurisdiction == "US")
            elif "eu_module" in name:
                param.requires_grad = (jurisdiction == "EU")
            elif "general_module" in name:
                param.requires_grad = (jurisdiction == "general")
            else:
                param.requires_grad = (jurisdiction == "general")
    
    def unfreeze_all(self):
        """Unfreeze all parameters (for general training phase)."""
        for param in self.parameters():
            param.requires_grad = True
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        jurisdiction: Optional[Jurisdiction] = None,
        return_dict: bool = True,
    ) -> Dict[str, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        
        if jurisdiction is None:
            jurisdiction = self.active_jurisdiction
        
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        x = self.token_embedding(input_ids) + self.position_embedding(pos_ids)
        x = self.dropout(x)
        
        for block in self.blocks:
            x = block(x, attention_mask)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        
        if not return_dict:
            return (logits, loss) if loss is not None else (logits,)
        
        return {
            "logits": logits,
            "loss": loss,
            "hidden_states": x,
        }
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        jurisdiction: Optional[Jurisdiction] = None,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate text with jurisdiction conditioning."""
        self.eval()
        
        if jurisdiction is not None:
            self.set_jurisdiction(jurisdiction)
        
        batch_size = input_ids.shape[0]
        
        for _ in range(max_new_tokens):
            if input_ids.shape[1] >= self.config.max_seq_len:
                input_ids = input_ids[:, -self.config.max_seq_len:]
                if attention_mask is not None:
                    attention_mask = attention_mask[:, -self.config.max_seq_len:]
            
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                jurisdiction=jurisdiction,
            )
            logits = outputs["logits"][:, -1, :]
            
            logits = logits / temperature
            
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float("-inf")
            
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float("-inf")
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((batch_size, 1), device=attention_mask.device, dtype=attention_mask.dtype)
                ], dim=-1)
            
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        
        return input_ids
    
    def get_jurisdiction_parameters(self, jurisdiction: Jurisdiction) -> int:
        """Count parameters for a specific jurisdiction (core + that jurisdiction modules)."""
        if jurisdiction == "core":
            return sum(p.numel() for n, p in self.named_parameters() 
                       if "us_module" not in n and "eu_module" not in n and "general_module" not in n)
        
        core_params = sum(p.numel() for n, p in self.named_parameters() 
                          if "us_module" not in n and "eu_module" not in n and "general_module" not in n)
        
        if jurisdiction == "US":
            module_params = sum(p.numel() for n, p in self.named_parameters() if "us_module" in n)
        elif jurisdiction == "EU":
            module_params = sum(p.numel() for n, p in self.named_parameters() if "eu_module" in n)
        elif jurisdiction == "general":
            module_params = sum(p.numel() for n, p in self.named_parameters() if "general_module" in n)
        else:
            module_params = 0
        
        return core_params + module_params
    
    def print_parameter_count(self):
        """Print parameter counts for core and each jurisdiction."""
        core_params = self.get_jurisdiction_parameters("core")
        print(f"Core parameters: {core_params:,} ({core_params/1e6:.2f}M)")
        
        for jur in self.config.jurisdictions:
            total = self.get_jurisdiction_parameters(jur)
            module_only = total - core_params
            print(f"{jur} total: {total:,} ({total/1e6:.2f}M) | module only: {module_only:,} ({module_only/1e6:.2f}M)")


def create_model(config: Optional[ModelConfig] = None) -> GRAMModel:
    """Factory function to create GRAM model."""
    if config is None:
        config = get_model_config()
    model = GRAMModel(config)
    model.print_parameter_count()
    return model


if __name__ == "__main__":
    config = get_model_config()
    model = create_model(config)
    
    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    print("\n=== Testing CORE forward ===")
    model.set_jurisdiction("core")
    out = model(input_ids, attention_mask, jurisdiction="core")
    print(f"Logits shape: {out['logits'].shape}")
    print(f"Loss: {out['loss']}")
    
    print("\n=== Testing US jurisdiction ===")
    model.set_jurisdiction("US")
    out = model(input_ids, attention_mask, jurisdiction="US")
    print(f"Logits shape: {out['logits'].shape}")
    print(f"Loss: {out['loss']}")
    
    print("\n=== Testing EU jurisdiction ===")
    model.set_jurisdiction("EU")
    out = model(input_ids, attention_mask, jurisdiction="EU")
    print(f"Logits shape: {out['logits'].shape}")
    print(f"Loss: {out['loss']}")
    
    print("\n=== Testing Full (US+EU) ===")
    model.set_module_config(enable_us=True, enable_eu=True)
    out = model(input_ids, attention_mask)
    print(f"Logits shape: {out['logits'].shape}")
    print(f"Loss: {out['loss']}")
    
    print("\n=== Testing generation ===")
    prompt = input_ids[:1, :10]
    generated = model.generate(prompt, max_new_tokens=20, jurisdiction="US")
    print(f"Generated shape: {generated.shape}")