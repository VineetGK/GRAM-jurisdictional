"""
GRAM Legal LLM Model
=====================

GRAM (Generic + Region-Adaptive Module) architecture:
- Shared core Transformer backbone (domain-general legal knowledge)
- Jurisdiction-specific adapter modules (US, EU) with LoRA-like low-rank adapters
- Gradient routing during training: core + jurisdiction-specific module
- Inference: select core + relevant jurisdiction module

Architecture:
```
Input -> Embedding -> Core Transformer Layers -> [Core + Jurisdiction Module] -> LM Head -> Logits
                                                         |
                                                         +-- US Adapter (LoRA)
                                                         +-- EU Adapter (LoRA)
                                                         +-- General Adapter (optional)
```

Based on:
- LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)
- AdapterFusion: Non-Destructive Task Composition for Transfer Learning (Pfeiffer et al., 2021)
- Modular Deep Learning (various)
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
    
    adapter_rank: int = 16
    adapter_alpha: float = 32.0
    adapter_dropout: float = 0.1
    
    jurisdictions: List[Jurisdiction] = None
    
    def __post_init__(self):
        if self.jurisdictions is None:
            self.jurisdictions = ["US", "EU", "general"]
        assert self.d_model % self.n_heads == 0


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
        adapter_rank=config.adapter_rank,
        adapter_alpha=config.adapter_alpha,
        adapter_dropout=config.adapter_dropout,
        jurisdictions=config.jurisdictions,
    )


class LoRAAdapter(nn.Module):
    """Low-Rank Adaptation module for jurisdiction-specific adaptation."""
    
    def __init__(
        self,
        d_model: int,
        rank: int,
        alpha: float,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        self.lora_A = nn.Linear(d_model, rank, bias=False)
        self.lora_B = nn.Linear(rank, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora_B(self.dropout(self.lora_A(x))) * self.scaling
    
    def merge_weights(self):
        """Merge LoRA weights into base layer (for inference optimization)."""
        return self.lora_B.weight @ self.lora_A.weight * self.scaling


class JurisdictionAdapter(nn.Module):
    """Adapter module for a specific jurisdiction (US, EU, General)."""
    
    def __init__(self, config: ModelConfig, jurisdiction: Jurisdiction):
        super().__init__()
        self.jurisdiction = jurisdiction
        self.config = config
        
        self.attention_adapter = LoRAAdapter(
            config.d_model, config.adapter_rank, config.adapter_alpha, config.adapter_dropout
        )
        self.ffn_adapter = LoRAAdapter(
            config.d_model, config.adapter_rank, config.adapter_alpha, config.adapter_dropout
        )
        self.layer_norm = LayerNorm(config.d_model, eps=config.layer_norm_eps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.layer_norm(x)
        x = self.attention_adapter(x) + self.ffn_adapter(x)
        return residual + x


class MultiHeadAttention(nn.Module):
    """Standard multi-head attention with optional LoRA adapters."""
    
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
        adapter: Optional[LoRAAdapter] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        
        if adapter is not None:
            q = q + adapter(q.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)).view(
                batch_size, seq_len, self.n_heads, self.d_head
            ).transpose(1, 2)
        
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


class FeedForward(nn.Module):
    """Feed-forward network with optional LoRA adapter."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.d_model, config.d_ff)
        self.fc2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = nn.GELU()
    
    def forward(
        self,
        x: torch.Tensor,
        adapter: Optional[LoRAAdapter] = None,
    ) -> torch.Tensor:
        residual = x
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        if adapter is not None:
            x = x + adapter(residual)
        
        return x


class TransformerBlock(nn.Module):
    """Single Transformer block with attention and FFN, supporting jurisdiction adapters."""
    
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        self.ln1 = LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.attention = MultiHeadAttention(config)
        self.ln2 = LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.ffn = FeedForward(config)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        jurisdiction_adapters: Optional[Dict[Jurisdiction, JurisdictionAdapter]] = None,
        active_jurisdiction: Optional[Jurisdiction] = None,
    ) -> torch.Tensor:
        adapter = None
        if jurisdiction_adapters and active_jurisdiction and active_jurisdiction in jurisdiction_adapters:
            adapter = jurisdiction_adapters[active_jurisdiction]
        
        attn_out = self.attention(
            self.ln1(x),
            attention_mask,
            adapter.attention_adapter if adapter else None,
        )
        x = x + self.dropout(attn_out)
        
        ffn_out = self.ffn(
            self.ln2(x),
            adapter.ffn_adapter if adapter else None,
        )
        x = x + self.dropout(ffn_out)
        
        return x


class GRAMModel(nn.Module):
    """
    GRAM (Generic + Region-Adaptive Module) Legal Language Model.
    
    Architecture:
    - Shared core Transformer backbone
    - Jurisdiction-specific LoRA adapters per layer
    - Gradient routing during training (core + active jurisdiction)
    - Inference: select core + relevant jurisdiction module
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
        
        self.jurisdiction_adapters = nn.ModuleDict({
            jur: nn.ModuleList([
                JurisdictionAdapter(config, jur) for _ in range(config.n_layers)
            ])
            for jur in config.jurisdictions
        })
        
        self.active_jurisdiction: Optional[Jurisdiction] = "core"
        self.training_mode: bool = True
        self.frozen_core: bool = False
        self.enable_us_module: bool = True
        self.enable_eu_module: bool = True
        
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
    
    def freeze_core(self, freeze: bool = True):
        """Freeze/unfreeze core model parameters."""
        self.frozen_core = freeze
        for name, param in self.named_parameters():
            if "jurisdiction_adapters" not in name:
                param.requires_grad = not freeze
    
    def get_trainable_parameters(self, jurisdiction: Optional[Jurisdiction] = None):
        """Get trainable parameters for a specific jurisdiction (core + jurisdiction adapters)."""
        if jurisdiction is None:
            jurisdiction = self.active_jurisdiction
        
        if jurisdiction == "core":
            return [p for n, p in self.named_parameters() if "jurisdiction_adapters" not in n]
        
        params = []
        for n, p in self.named_parameters():
            if "jurisdiction_adapters" not in n:
                params.append(p)
            elif f"jurisdiction_adapters.{jurisdiction}" in n:
                params.append(p)
        return params
    
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
        
        adapters = None
        if jurisdiction != "core":
            # Only include enabled modules
            enabled_jurisdictions = []
            if self.enable_us_module and "US" in self.config.jurisdictions:
                enabled_jurisdictions.append("US")
            if self.enable_eu_module and "EU" in self.config.jurisdictions:
                enabled_jurisdictions.append("EU")
            if "general" in self.config.jurisdictions:
                enabled_jurisdictions.append("general")
            
            adapters = {
                jur: self.jurisdiction_adapters[jur] for jur in enabled_jurisdictions
            }
        
        for i, block in enumerate(self.blocks):
            active_adapter = None
            if adapters and jurisdiction != "core" and jurisdiction in adapters:
                active_adapter = {jurisdiction: adapters[jurisdiction][i]}
            
            x = block(
                x,
                attention_mask,
                active_adapter,
                jurisdiction if jurisdiction != "core" else None,
            )
        
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
    
    def merge_adapters(self, jurisdiction: Jurisdiction):
        """Merge jurisdiction adapters into core weights for faster inference."""
        if jurisdiction == "core":
            return
        
        adapters = self.jurisdiction_adapters[jurisdiction]
        
        for i, block in enumerate(self.blocks):
            adapter = adapters[i]
            
            lora_a_weight = adapter.attention_adapter.lora_A.weight
            lora_b_weight = adapter.attention_adapter.lora_B.weight
            scaling = adapter.attention_adapter.scaling
            merged = lora_b_weight @ lora_a_weight * scaling
            
            block.attention.out_proj.weight.data += merged.T
            
            lora_a_weight = adapter.ffn_adapter.lora_A.weight
            lora_b_weight = adapter.ffn_adapter.lora_B.weight
            scaling = adapter.ffn_adapter.scaling
            merged = lora_b_weight @ lora_a_weight * scaling
            
            block.ffn.fc2.weight.data += merged.T
        
        print(f"Merged {jurisdiction} adapters into core model")
    
    def unmerge_adapters(self, jurisdiction: Jurisdiction):
        """Unmerge adapters (requires keeping original weights)."""
        raise NotImplementedError("Unmerge requires storing original weights")
    
    def get_jurisdiction_parameters(self, jurisdiction: Jurisdiction) -> int:
        """Count parameters for a specific jurisdiction (core + adapters)."""
        if jurisdiction == "core":
            return sum(p.numel() for n, p in self.named_parameters() 
                       if "jurisdiction_adapters" not in n)
        
        core_params = sum(p.numel() for n, p in self.named_parameters() 
                          if "jurisdiction_adapters" not in n)
        adapter_params = sum(p.numel() for n, p in self.named_parameters()
                             if f"jurisdiction_adapters.{jurisdiction}" in n)
        return core_params + adapter_params
    
    def print_parameter_count(self):
        """Print parameter counts for core and each jurisdiction."""
        core_params = self.get_jurisdiction_parameters("core")
        print(f"Core parameters: {core_params:,} ({core_params/1e6:.2f}M)")
        
        for jur in self.config.jurisdictions:
            total = self.get_jurisdiction_parameters(jur)
            adapter_only = total - core_params
            print(f"{jur} total: {total:,} ({total/1e6:.2f}M) | adapters only: {adapter_only:,} ({adapter_only/1e6:.2f}M)")


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
    
    print("\n=== Testing generation ===")
    prompt = input_ids[:1, :10]
    generated = model.generate(prompt, max_new_tokens=20, jurisdiction="US")
    print(f"Generated shape: {generated.shape}")