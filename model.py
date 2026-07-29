"""GRAM-style Transformer model with modular auxiliary modules for US/EU jurisdictions."""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from config import config


@dataclass
class ModelConfig:
    vocab_size: int = config.vocab_size
    max_seq_len: int = config.max_seq_len
    n_layers: int = config.n_layers
    n_heads: int = config.n_heads
    n_embd: int = config.n_embd
    n_inner: int = config.n_inner
    dropout: float = config.dropout
    attn_dropout: float = config.attn_dropout
    resid_dropout: float = config.resid_dropout
    embd_pdrop: float = config.embd_pdrop
    layer_norm_epsilon: float = config.layer_norm_epsilon
    initializer_range: float = config.initializer_range
    enable_us_module: bool = config.enable_us_module
    enable_eu_module: bool = config.enable_eu_module
    module_mlp_ratio: float = config.module_mlp_ratio

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_heads

    @property
    def module_inner(self) -> int:
        return int(self.n_inner * self.module_mlp_ratio)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with flash attention support."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_heads == 0
        self.cfg = cfg

        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)

        self.attn_dropout = nn.Dropout(cfg.attn_dropout)
        self.resid_dropout = nn.Dropout(cfg.resid_dropout)

        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.scale = self.head_dim ** -0.5

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len))
            .view(1, 1, cfg.max_seq_len, cfg.max_seq_len),
            persistent=False,
        )

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.cfg.n_embd, dim=2)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * self.scale

        causal_mask = self.bias[:, :, :T, :T]
        att = att.masked_fill(causal_mask == 0, float('-inf'))

        if attention_mask is not None:
            att = att + (1.0 - attention_mask[:, None, None, :]) * -1e9

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_dropout(self.c_proj(y))
        return y


class GPTMLP(nn.Module):
    """Standard GPT-style MLP (GeGLU or GELU)."""

    def __init__(self, cfg: ModelConfig, inner_dim: int = None):
        super().__init__()
        inner = inner_dim or cfg.n_inner
        self.c_fc = nn.Linear(cfg.n_embd, inner, bias=False)
        self.c_proj = nn.Linear(inner, cfg.n_embd, bias=False)
        self.dropout = nn.Dropout(cfg.resid_dropout)
        self.act = nn.GELU(approximate='tanh')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class GRAMBlock(nn.Module):
    """
    Transformer block with GRAM-style modular auxiliary MLPs.
    Core MLP is always active. US/EU MLPs are conditionally active based on flags.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.ln_1 = nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_epsilon)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_epsilon)

        # Core MLP (always active)
        self.core_mlp = GPTMLP(cfg)

        # Auxiliary modules
        self.enable_us_module = cfg.enable_us_module
        self.enable_eu_module = cfg.enable_eu_module

        if self.enable_us_module:
            self.us_mlp = GPTMLP(cfg, inner_dim=cfg.module_inner)
            self.us_ln = nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_epsilon)

        if self.enable_eu_module:
            self.eu_mlp = GPTMLP(cfg, inner_dim=cfg.module_inner)
            self.eu_ln = nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_epsilon)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        enable_us: bool = True,
        enable_eu: bool = True,
    ) -> torch.Tensor:
        # Self-attention (always active)
        residual = x
        x = self.ln_1(x)
        x = self.attn(x, attention_mask)
        x = residual + x

        # Core MLP (always active)
        residual = x
        x = self.ln_2(x)
        core_out = self.core_mlp(x)

        # Module outputs (conditionally active)
        module_out = torch.zeros_like(core_out)

        if self.enable_us_module and enable_us:
            us_out = self.us_mlp(self.us_ln(x))
            module_out = module_out + us_out

        if self.enable_eu_module and enable_eu:
            eu_out = self.eu_mlp(self.eu_ln(x))
            module_out = module_out + eu_out

        x = residual + core_out + module_out
        return x

    def set_module_grad(self, us_grad: bool, eu_grad: bool, core_grad: bool = True):
        """Set requires_grad for module parameters (for GRAM routing)."""
        for p in self.core_mlp.parameters():
            p.requires_grad = core_grad

        if self.enable_us_module:
            for p in self.us_mlp.parameters():
                p.requires_grad = us_grad
            for p in self.us_ln.parameters():
                p.requires_grad = us_grad

        if self.enable_eu_module:
            for p in self.eu_mlp.parameters():
                p.requires_grad = eu_grad
            for p in self.eu_ln.parameters():
                p.requires_grad = eu_grad


class GRAMTransformer(nn.Module):
    """
    GRAM-style decoder-only Transformer with jurisdiction-aware modules.
    Supports runtime configuration of enabled modules.
    """

    def __init__(self, cfg: ModelConfig = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        self.enable_us_module = self.cfg.enable_us_module
        self.enable_eu_module = self.cfg.enable_eu_module

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(self.cfg.vocab_size, self.cfg.n_embd),
            wpe=nn.Embedding(self.cfg.max_seq_len, self.cfg.n_embd),
            drop=nn.Dropout(self.cfg.embd_pdrop),
            h=nn.ModuleList([GRAMBlock(self.cfg) for _ in range(self.cfg.n_layers)]),
            ln_f=nn.LayerNorm(self.cfg.n_embd, eps=self.cfg.layer_norm_epsilon),
        ))

        self.lm_head = nn.Linear(self.cfg.n_embd, self.cfg.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.cfg.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.cfg.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        enable_us_module: bool = None,
        enable_eu_module: bool = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional module control.
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            labels: (batch_size, seq_len) for loss computation
            enable_us_module: override US module flag
            enable_eu_module: override EU module flag
        """
        us_enabled = self.enable_us_module if enable_us_module is None else enable_us_module
        eu_enabled = self.enable_eu_module if enable_eu_module is None else enable_eu_module

        B, T = input_ids.shape
        assert T <= self.cfg.max_seq_len, f"Sequence length {T} exceeds max {self.cfg.max_seq_len}"

        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device).unsqueeze(0)

        tok_emb = self.transformer.wte(input_ids)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x, attention_mask, enable_us=us_enabled, enable_eu=eu_enabled)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        return {
            "logits": logits,
            "loss": loss,
            "hidden_states": x,
        }

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
        enable_us_module: bool = None,
        enable_eu_module: bool = None,
        eos_token_id: int = None,
        pad_token_id: int = None,
    ) -> torch.Tensor:
        """Generate text using the model."""
        self.eval()
        us_enabled = self.enable_us_module if enable_us_module is None else enable_us_module
        eu_enabled = self.enable_eu_module if enable_eu_module is None else enable_eu_module

        generated = input_ids.clone()
        past_key_values = None

        for _ in range(max_new_tokens):
            # Crop context if too long
            if generated.shape[1] > self.cfg.max_seq_len:
                generated = generated[:, -self.cfg.max_seq_len:]

            with torch.no_grad():
                outputs = self.forward(
                    generated,
                    enable_us_module=us_enabled,
                    enable_eu_module=eu_enabled,
                )
                logits = outputs["logits"][:, -1, :] / temperature

            # Repetition penalty
            if repetition_penalty != 1.0:
                for i in range(generated.shape[0]):
                    for token_id in set(generated[i].tolist()):
                        logits[i, token_id] /= repetition_penalty

            # Top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = -float('inf')

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return generated

    def set_module_grad(self, us_grad: bool, eu_grad: bool, core_grad: bool = True):
        """Set requires_grad for all blocks (for GRAM gradient routing)."""
        for block in self.transformer.h:
            block.set_module_grad(us_grad, eu_grad, core_grad)

    def get_parameter_groups(self) -> Dict[str, List[nn.Parameter]]:
        """Get parameter groups for optimizer (core, us, eu)."""
        groups = {"core": [], "us": [], "eu": []}

        # Embedding and output layers go to core
        for p in self.transformer.wte.parameters():
            groups["core"].append(p)
        for p in self.transformer.wpe.parameters():
            groups["core"].append(p)
        for p in self.transformer.ln_f.parameters():
            groups["core"].append(p)
        for p in self.lm_head.parameters():
            groups["core"].append(p)

        for block in self.transformer.h:
            for p in block.ln_1.parameters():
                groups["core"].append(p)
            for p in block.attn.parameters():
                groups["core"].append(p)
            for p in block.ln_2.parameters():
                groups["core"].append(p)
            for p in block.core_mlp.parameters():
                groups["core"].append(p)

            if self.enable_us_module:
                for p in block.us_mlp.parameters():
                    groups["us"].append(p)
                for p in block.us_ln.parameters():
                    groups["us"].append(p)

            if self.enable_eu_module:
                for p in block.eu_mlp.parameters():
                    groups["eu"].append(p)
                for p in block.eu_ln.parameters():
                    groups["eu"].append(p)

        return groups

    def set_config(self, enable_us_module: bool = None, enable_eu_module: bool = None):
        """Update module enable flags at runtime."""
        if enable_us_module is not None:
            self.enable_us_module = enable_us_module
        if enable_eu_module is not None:
            self.enable_eu_module = enable_eu_module

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def num_parameters_per_module(self) -> Dict[str, int]:
        groups = self.get_parameter_groups()
        return {k: sum(p.numel() for p in v) for k, v in groups.items()}


def create_model(cfg: ModelConfig = None) -> GRAMTransformer:
    """Factory function to create GRAM model."""
    return GRAMTransformer(cfg)


if __name__ == "__main__":
    # Test model
    print("Creating GRAM model...")
    model = create_model()
    print(f"Total parameters: {model.num_parameters():,}")
    print(f"Parameters per module: {model.num_parameters_per_module()}")

    # Test forward
    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    print("\nTesting forward pass (Full)...")
    out = model(input_ids, attention_mask, enable_us_module=True, enable_eu_module=True)
    print(f"Logits shape: {out['logits'].shape}")
    print(f"Loss: {out['loss']}")

    print("\nTesting forward pass (US-only)...")
    out = model(input_ids, attention_mask, enable_us_module=True, enable_eu_module=False)
    print(f"Logits shape: {out['logits'].shape}")

    print("\nTesting forward pass (EU-only)...")
    out = model(input_ids, attention_mask, enable_us_module=False, enable_eu_module=True)
    print(f"Logits shape: {out['logits'].shape}")

    print("\nTesting generate...")
    generated = model.generate(input_ids[:, :10], max_new_tokens=20, enable_us_module=True, enable_eu_module=True)
    print(f"Generated shape: {generated.shape}")

    print("\nTesting gradient routing...")
    model.set_module_grad(us_grad=True, eu_grad=False, core_grad=False)
    us_params_grad = sum(p.requires_grad for p in model.parameters() if any(p in g for g in model.get_parameter_groups()["us"]))
    print(f"US params with grad: {us_params_grad}")

    model.set_module_grad(us_grad=False, eu_grad=True, core_grad=False)
    eu_params_grad = sum(p.requires_grad for p in model.parameters() if any(p in g for g in model.get_parameter_groups()["eu"]))
    print(f"EU params with grad: {eu_params_grad}")

    print("\nAll tests passed!")