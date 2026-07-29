"""
GRAM Training Loop
===================

Implements GRAM (Generic + Region-Adaptive Module) training:
- Stage 1: Pre-train core + all adapters jointly on mixed corpus
- Stage 2 (optional): Fine-tune individual jurisdiction adapters

Gradient Routing:
- For US samples: gradients flow to core + US adapter
- For EU samples: gradients flow to core + EU adapter  
- For General samples: gradients flow to core only (or core + general adapter)

This enables parameter-efficient multi-jurisdiction learning.
"""

import os
import math
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from tqdm import tqdm

from config import config
from model import GRAMModel, ModelConfig
from tokenizer import load_tokenizer, train_tokenizer
from datasets import create_unified_dataloader


@dataclass
class TrainingConfig:
    num_epochs: int = 3
    max_steps: int = -1
    warmup_steps: int = 500
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    
    use_amp: bool = True
    compile_model: bool = False
    
    log_interval: int = 50
    eval_interval: int = 500
    save_interval: int = 1000
    
    eval_batches: int = 50
    
    jurisdiction_schedule: str = "mixed"  # "mixed", "sequential", "alternating"
    core_lr_multiplier: float = 1.0
    adapter_lr_multiplier: float = 1.0
    
    gradient_routing: str = "jurisdiction"  # "jurisdiction", "all", "core_only"


train_config = TrainingConfig()


class GRAMTrainer:
    def __init__(
        self,
        model: GRAMModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        tokenizer,
        config: TrainingConfig = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.config = config or train_config
        
        self.device = config.device if hasattr(config, 'device') else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        if self.config.compile_model:
            self.model = torch.compile(self.model)
        
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.scaler = GradScaler(enabled=self.config.use_amp)
        
        self.step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        config.log_dir.mkdir(parents=True, exist_ok=True)
    
    def _create_optimizer(self) -> AdamW:
        core_params = []
        adapter_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "adapter" in name.lower() or "lora" in name.lower():
                adapter_params.append(param)
            elif "core" in name.lower() or "transformer" in name.lower() or "embed" in name.lower() or "ln_f" in name.lower() or "lm_head" in name.lower():
                core_params.append(param)
            else:
                other_params.append(param)
        
        param_groups = [
            {"params": core_params, "lr": self.config.learning_rate * self.config.core_lr_multiplier, "weight_decay": self.config.weight_decay},
            {"params": adapter_params, "lr": self.config.learning_rate * self.config.adapter_lr_multiplier, "weight_decay": self.config.weight_decay},
            {"params": other_params, "lr": self.config.learning_rate, "weight_decay": self.config.weight_decay},
        ]
        
        param_groups = [g for g in param_groups if len(g["params"]) > 0]
        
        print(f"Optimizer param groups: core={len(core_params)}, adapter={len(adapter_params)}, other={len(other_params)}")
        
        return AdamW(
            param_groups,
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
            eps=self.config.eps,
        )
    
    def _create_scheduler(self):
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / max(1, self.config.warmup_steps)
            
            if self.config.max_steps > 0:
                progress = (step - self.config.warmup_steps) / max(1, self.config.max_steps - self.config.warmup_steps)
            else:
                progress = 0
            
            return max(
                self.config.min_lr / self.config.learning_rate,
                0.5 * (1 + math.cos(math.pi * progress))
            )
        
        return LambdaLR(self.optimizer, lr_lambda)
    
    def _route_gradients(self, batch: Dict, jurisdiction: str):
        if self.config.gradient_routing == "all":
            pass
        elif self.config.gradient_routing == "core_only":
            for name, param in self.model.named_parameters():
                if "adapter" in name.lower() or "lora" in name.lower():
                    if param.grad is not None:
                        param.grad.zero_()
        elif self.config.gradient_routing == "jurisdiction":
            for name, param in self.model.named_parameters():
                if "adapter" in name.lower() or "lora" in name.lower():
                    adapter_jurisdiction = None
                    if "us" in name.lower():
                        adapter_jurisdiction = "US"
                    elif "eu" in name.lower():
                        adapter_jurisdiction = "EU"
                    elif "general" in name.lower():
                        adapter_jurisdiction = "general"
                    
                    if adapter_jurisdiction and adapter_jurisdiction != jurisdiction:
                        if param.grad is not None:
                            param.grad.zero_()
    
    def train_step(self, batch: Dict) -> Dict:
        self.model.train()
        
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)
        jurisdictions = batch["jurisdictions"]
        
        jurisdiction = jurisdictions[0] if jurisdictions else "general"
        
        with autocast(enabled=self.config.use_amp):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                jurisdiction=jurisdiction,
            )
            loss = outputs["loss"]
            loss = loss / self.config.gradient_accumulation_steps
        
        self.scaler.scale(loss).backward()
        
        self._route_gradients(batch, jurisdiction)
        
        return {
            "loss": loss.item() * self.config.gradient_accumulation_steps,
            "jurisdiction": jurisdiction,
        }
    
    def train_epoch(self) -> Dict:
        self.model.train()
        total_loss = 0
        jurisdiction_losses = {"US": 0, "EU": 0, "general": 0}
        jurisdiction_counts = {"US": 0, "EU": 0, "general": 0}
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            step_metrics = self.train_step(batch)
            total_loss += step_metrics["loss"]
            jur = step_metrics["jurisdiction"]
            jurisdiction_losses[jur] += step_metrics["loss"]
            jurisdiction_counts[jur] += 1
            
            self.step += 1
            
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()
                
                if self.step % self.config.log_interval == 0:
                    avg_loss = total_loss / (batch_idx + 1)
                    lr = self.optimizer.param_groups[0]["lr"]
                    pbar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{lr:.2e}", "step": self.step})
                    
                    self.train_losses.append(avg_loss)
                    self.learning_rates.append(lr)
                
                if self.step % self.config.eval_interval == 0:
                    val_metrics = self.evaluate()
                    self.val_losses.append(val_metrics["val_loss"])
                    self.model.train()
                    
                    if val_metrics["val_loss"] < self.best_val_loss:
                        self.best_val_loss = val_metrics["val_loss"]
                        self.save_checkpoint("best")
                
                if self.step % self.config.save_interval == 0:
                    self.save_checkpoint(f"step_{self.step}")
                
                if self.config.max_steps > 0 and self.step >= self.config.max_steps:
                    break
        
        avg_loss = total_loss / len(self.train_loader)
        return {
            "train_loss": avg_loss,
            "jurisdiction_losses": {k: v / max(1, jurisdiction_counts[k]) for k, v in jurisdiction_losses.items()},
        }
    
    @torch.no_grad()
    def evaluate(self) -> Dict:
        self.model.eval()
        total_loss = 0
        jurisdiction_losses = {"US": 0, "EU": 0, "general": 0}
        jurisdiction_counts = {"US": 0, "EU": 0, "general": 0}
        
        eval_batches = min(self.config.eval_batches, len(self.val_loader))
        
        for batch_idx, batch in enumerate(self.val_loader):
            if batch_idx >= eval_batches:
                break
            
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            jurisdictions = batch["jurisdictions"]
            
            jurisdiction = jurisdictions[0] if jurisdictions else "general"
            
            with autocast(enabled=self.config.use_amp):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    jurisdiction=jurisdiction,
                )
                loss = outputs["loss"]
            
            total_loss += loss.item()
            jurisdiction_losses[jurisdiction] += loss.item()
            jurisdiction_counts[jurisdiction] += 1
        
        avg_loss = total_loss / max(1, eval_batches)
        
        print(f"\nValidation Loss: {avg_loss:.4f}")
        for jur in jurisdiction_losses:
            if jurisdiction_counts[jur] > 0:
                print(f"  {jur}: {jurisdiction_losses[jur] / jurisdiction_counts[jur]:.4f}")
        
        return {
            "val_loss": avg_loss,
            "jurisdiction_losses": {k: v / max(1, jurisdiction_counts[k]) for k, v in jurisdiction_losses.items()},
        }
    
    def save_checkpoint(self, name: str):
        checkpoint = {
            "step": self.step,
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "learning_rates": self.learning_rates,
            "config": asdict(self.config) if hasattr(self.config, '__dataclass_fields__') else {},
            "model_config": asdict(self.model.config) if hasattr(self.model.config, '__dataclass_fields__') else {},
        }
        
        path = config.checkpoint_dir / f"{name}.pt"
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")
        
        latest_path = config.checkpoint_dir / "latest.pt"
        torch.save(checkpoint, latest_path)
    
    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        self.step = checkpoint["step"]
        self.epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.train_losses = checkpoint.get("train_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])
        self.learning_rates = checkpoint.get("learning_rates", [])
        
        print(f"Loaded checkpoint from {path} (step {self.step}, epoch {self.epoch})")
    
    def train(self):
        print(f"Starting training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            epoch_metrics = self.train_epoch()
            
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {epoch_metrics['train_loss']:.4f}")
            for jur, loss in epoch_metrics['jurisdiction_losses'].items():
                print(f"  {jur} Loss: {loss:.4f}")
            
            val_metrics = self.evaluate()
            
            if val_metrics["val_loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["val_loss"]
                self.save_checkpoint("best")
            
            self.save_checkpoint(f"epoch_{epoch}")
            
            if self.config.max_steps > 0 and self.step >= self.config.max_steps:
                break
        
        print("Training complete!")
        self.save_checkpoint("final")


def create_model_and_tokenizer() -> Tuple[GRAMModel, object]:
    tokenizer_path = config.tokenizer_dir / "tokenizer.json"
    if not tokenizer_path.exists():
        print("Tokenizer not found. Training new tokenizer...")
        train_tokenizer(
            data_dir=str(config.data_dir),
            vocab_size=config.tokenizer_vocab_size,
            min_frequency=config.tokenizer_min_frequency,
            output_dir=str(config.tokenizer_dir),
        )
    
    tokenizer = load_tokenizer(str(config.tokenizer_dir))
    
    model_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=config.max_seq_len,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        dropout=config.dropout,
        adapter_rank=config.adapter_rank,
        adapter_alpha=config.adapter_alpha,
    )
    
    model = GRAMModel(model_config)
    
    return model, tokenizer


def main():
    print("=" * 60)
    print("GRAM Legal LLM Training")
    print("=" * 60)
    
    model, tokenizer = create_model_and_tokenizer()
    
    print("Creating dataloaders...")
    train_loader, val_loader = create_unified_dataloader(
        tokenizer,
        batch_size=config.batch_size,
        max_seq_len=config.max_seq_len,
        chunk_size=config.chunk_size,
        num_workers=config.num_workers,
    )
    
    trainer = GRAMTrainer(model, train_loader, val_loader, tokenizer)
    
    latest_checkpoint = config.checkpoint_dir / "latest.pt"
    if latest_checkpoint.exists():
        trainer.load_checkpoint(str(latest_checkpoint))
    
    trainer.train()


if __name__ == "__main__":
    main()