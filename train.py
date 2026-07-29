"""Training script for GRAM legal LLM with gradient-routed auxiliary modules."""

import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from config import config
from model import GRAMTransformer, ModelConfig, create_model
from tokenizer import LegalBPETokenizer, get_tokenizer, train_tokenizer_from_datasets
from data import (
    create_dataloaders,
    MixedJurisdictionDataLoader,
    JurisdictionBatch,
    collate_batch,
    get_dataloaders,
)


class GRAMTrainer:
    """Trainer with GRAM gradient routing for jurisdiction-aware modules."""

    def __init__(
        self,
        model: GRAMTransformer,
        train_loader: MixedJurisdictionDataLoader,
        val_us_loader: DataLoader,
        val_eu_loader: DataLoader,
        tokenizer: LegalBPETokenizer,
        cfg: config = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_us_loader = val_us_loader
        self.val_eu_loader = val_eu_loader
        self.tokenizer = tokenizer
        self.cfg = cfg or config

        self.device = torch.device(self.cfg.device)
        self.dtype = self.cfg.get_dtype()
        self.model.to(self.device)

        # Optimizer with parameter groups for GRAM
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # Training state
        self.step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")
        self.phase = "warmup"  # "warmup" or "gram"

        # Logging
        self.log_dir = self.cfg.logs_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.cfg.checkpoints_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Gradient scaling for mixed precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.dtype == torch.float16)

    def _create_optimizer(self) -> AdamW:
        """Create optimizer with parameter groups for GRAM modules."""
        param_groups = self.model.get_parameter_groups()

        groups = [
            {"params": param_groups["core"], "lr": self.cfg.learning_rate, "name": "core"},
            {"params": param_groups["us"], "lr": self.cfg.learning_rate, "name": "us_module"},
            {"params": param_groups["eu"], "lr": self.cfg.learning_rate, "name": "eu_module"},
        ]

        # Filter empty groups
        groups = [g for g in groups if len(g["params"]) > 0]

        print(f"Optimizer groups:")
        for g in groups:
            n_params = sum(p.numel() for p in g["params"])
            print(f"  {g['name']}: {n_params:,} params")

        return AdamW(
            groups,
            lr=self.cfg.learning_rate,
            betas=(self.cfg.beta1, self.cfg.beta2),
            eps=self.cfg.eps,
            weight_decay=self.cfg.weight_decay,
        )

    def _create_scheduler(self):
        """Create learning rate scheduler."""
        if self.cfg.lr_schedule == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.cfg.max_steps,
                eta_min=self.cfg.min_lr,
            )
        elif self.cfg.lr_schedule == "linear":
            return torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=self.cfg.min_lr / self.cfg.learning_rate,
                total_iters=self.cfg.max_steps,
            )
        else:
            return torch.optim.lr_scheduler.ConstantLR(self.optimizer, factor=1.0)

    def _get_grad_routing_flags(self, jurisdiction: torch.Tensor) -> Tuple[bool, bool, bool]:
        """
        Determine which modules should receive gradients based on jurisdiction.
        Returns: (core_grad, us_grad, eu_grad)
        """
        if self.phase == "warmup":
            # Phase 1: general warm-up - update core only (or all)
            if self.cfg.general_updates_core_only:
                return True, False, False
            return True, True, True

        # Phase 2: GRAM routing
        # jurisdiction: 0=general, 1=US, 2=EU
        jur = jurisdiction.item() if jurisdiction.numel() == 1 else jurisdiction[0].item()

        if jur == 1:  # US batch
            return (
                not self.cfg.freeze_core_during_gram,
                True,
                not self.cfg.freeze_other_module_during_gram,
            )
        elif jur == 2:  # EU batch
            return (
                not self.cfg.freeze_core_during_gram,
                not self.cfg.freeze_other_module_during_gram,
                True,
            )
        else:  # general batch
            if self.cfg.general_updates_core_only:
                return True, False, False
            return True, True, True

    def _apply_grad_routing(self, batch: JurisdictionBatch):
        """Apply GRAM gradient routing based on batch jurisdiction."""
        core_grad, us_grad, eu_grad = self._get_grad_routing_flags(batch.jurisdiction)

        # Apply to model
        self.model.set_module_grad(us_grad=us_grad, eu_grad=eu_grad, core_grad=core_grad)

        # Also apply to optimizer param groups
        for group in self.optimizer.param_groups:
            if group["name"] == "core":
                for p in group["params"]:
                    p.requires_grad = core_grad
            elif group["name"] == "us_module":
                for p in group["params"]:
                    p.requires_grad = us_grad
            elif group["name"] == "eu_module":
                for p in group["params"]:
                    p.requires_grad = eu_grad

    def _train_step(self, batch: JurisdictionBatch) -> Dict[str, float]:
        """Single training step with GRAM gradient routing."""
        self.model.train()

        # Apply gradient routing
        self._apply_grad_routing(batch)

        input_ids = batch.input_ids.to(self.device)
        attention_mask = batch.attention_mask.to(self.device)
        labels = batch.labels.to(self.device)

        # Determine module enables for forward pass
        jur = batch.jurisdiction[0].item() if batch.jurisdiction.numel() > 0 else 0
        enable_us = (jur == 1) or self.phase == "warmup"
        enable_eu = (jur == 2) or self.phase == "warmup"

        with torch.autocast(device_type=self.device.type, dtype=self.dtype):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                enable_us_module=enable_us,
                enable_eu_module=enable_eu,
            )
            loss = outputs["loss"] / self.cfg.gradient_accumulation_steps

        self.scaler.scale(loss).backward()

        return {
            "loss": loss.item() * self.cfg.gradient_accumulation_steps,
            "jurisdiction": jur,
            "enable_us": enable_us,
            "enable_eu": enable_eu,
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation on US and EU datasets."""
        self.model.eval()
        results = {}

        for name, loader in [("us", self.val_us_loader), ("eu", self.val_eu_loader)]:
            total_loss = 0.0
            total_tokens = 0
            count = 0

            for batch in loader:
                if count >= self.cfg.eval_steps:
                    break

                input_ids = batch.input_ids.to(self.device)
                attention_mask = batch.attention_mask.to(self.device)
                labels = batch.labels.to(self.device)

                with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        enable_us_module=True,
                        enable_eu_module=True,
                    )
                    loss = outputs["loss"]

                total_loss += loss.item() * input_ids.numel()
                total_tokens += input_ids.numel()
                count += 1

            avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
            ppl = math.exp(min(avg_loss, 20))

            results[f"val_{name}_loss"] = avg_loss
            results[f"val_{name}_ppl"] = ppl

        return results

    def save_checkpoint(self, name: str = None, is_best: bool = False):
        """Save model checkpoint."""
        if name is None:
            name = f"checkpoint_step_{self.step}.pt"

        path = self.checkpoint_dir / name
        best_path = self.checkpoint_dir / "best_model.pt"
        latest_path = self.checkpoint_dir / "latest_model.pt"

        checkpoint = {
            "step": self.step,
            "epoch": self.epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "phase": self.phase,
            "config": {
                "model": self.model.cfg.__dict__,
                "train": self.cfg.__dict__,
            },
        }

        torch.save(checkpoint, path)
        torch.save(checkpoint, latest_path)

        if is_best:
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")

        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.scaler.load_state_dict(checkpoint["scaler"])
        self.step = checkpoint["step"]
        self.epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.phase = checkpoint.get("phase", "warmup")
        print(f"Loaded checkpoint from {path} (step {self.step})")

    def train(self):
        """Main training loop with GRAM phases."""
        print("=" * 60)
        print("Starting GRAM Training")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Dtype: {self.dtype}")
        print(f"Max steps: {self.cfg.max_steps}")
        print(f"Warmup steps: {self.cfg.warmup_steps}")
        print(f"Phase 1 (warmup): {self.cfg.phase1_general_warmup_steps} steps")
        print(f"Phase 2 (GRAM): {self.cfg.phase2_gram_steps} steps")
        print(f"Gradient accumulation: {self.cfg.gradient_accumulation_steps}")
        print()

        self.model.train()
        accumulation_step = 0
        epoch_loss = 0.0
        epoch_steps = 0
        log_losses = []

        # Initial phase
        self.train_loader.set_phase("warmup")
        self.phase = "warmup"

        try:
            while self.step < self.cfg.max_steps:
                # Switch to GRAM phase
                if self.step == self.cfg.phase1_general_warmup_steps:
                    print("\n" + "=" * 60)
                    print("SWITCHING TO GRAM PHASE (Phase 2)")
                    print("=" * 60)
                    self.train_loader.set_phase("gram")
                    self.phase = "gram"

                for batch in self.train_loader:
                    if self.step >= self.cfg.max_steps:
                        break

                    # Training step
                    step_result = self._train_step(batch)
                    epoch_loss += step_result["loss"]
                    epoch_steps += 1
                    accumulation_step += 1
                    log_losses.append(step_result["loss"])

                    # Optimizer step
                    if accumulation_step >= self.cfg.gradient_accumulation_steps:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.cfg.max_grad_norm,
                        )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.scheduler.step()
                        self.optimizer.zero_grad()
                        accumulation_step = 0
                        self.step += 1

                        # Logging
                        if self.step % self.cfg.log_interval == 0:
                            avg_loss = sum(log_losses) / len(log_losses)
                            lr = self.optimizer.param_groups[0]["lr"]
                            print(
                                f"Step {self.step}/{self.cfg.max_steps} | "
                                f"Phase: {self.phase} | "
                                f"Loss: {avg_loss:.4f} | "
                                f"LR: {lr:.2e}"
                            )
                            log_losses = []

                        # Evaluation
                        if self.step % self.cfg.eval_interval == 0:
                            val_results = self.validate()
                            print(
                                f"  Val US: loss={val_results['val_us_loss']:.4f}, "
                                f"ppl={val_results['val_us_ppl']:.2f} | "
                                f"Val EU: loss={val_results['val_eu_loss']:.4f}, "
                                f"ppl={val_results['val_eu_ppl']:.2f}"
                            )

                            # Save best model
                            val_loss = (val_results["val_us_loss"] + val_results["val_eu_loss"]) / 2
                            if val_loss < self.best_val_loss:
                                self.best_val_loss = val_loss
                                self.save_checkpoint(is_best=True)

                        # Checkpoint
                        if self.step % self.cfg.save_interval == 0:
                            self.save_checkpoint()

                    # Break inner loop if max steps reached
                    if self.step >= self.cfg.max_steps:
                        break

                self.epoch += 1

        except KeyboardInterrupt:
            print("\nTraining interrupted, saving checkpoint...")
            self.save_checkpoint(name="interrupted_model.pt")

        # Final save
        self.save_checkpoint(name="final_model.pt")
        print("\nTraining complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")


def setup_tokenizer() -> LegalBPETokenizer:
    """Setup or train tokenizer."""
    tokenizer_path = config.tokenizer_dir / "tokenizer.json"

    if tokenizer_path.exists():
        print(f"Loading existing tokenizer from {tokenizer_path}")
        return LegalBPETokenizer.load(str(tokenizer_path))

    print("Training new tokenizer...")
    tokenizer = train_tokenizer_from_datasets(
        us_data_path=str(config.us_data_path) if config.us_data_path.exists() else None,
        eu_data_path=str(config.eu_data_path) if config.eu_data_path.exists() else None,
        general_data_path=str(config.general_data_path) if config.general_data_path.exists() else None,
        vocab_size=config.vocab_size,
        output_path=str(tokenizer_path),
    )
    return tokenizer


def main():
    """Main training entry point."""
    # Set seed
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # Setup tokenizer
    tokenizer = setup_tokenizer()
    print(f"Tokenizer vocab size: {len(tokenizer)}")

    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_us_loader, val_eu_loader = get_dataloaders(tokenizer)

    # Create model
    print("Creating model...")
    model_cfg = ModelConfig(
        vocab_size=len(tokenizer),
        enable_us_module=True,
        enable_eu_module=True,
    )
    model = create_model(model_cfg)
    print(f"Model parameters: {model.num_parameters():,}")
    print(f"Per module: {model.num_parameters_per_module()}")

    # Create trainer
    trainer = GRAMTrainer(
        model=model,
        train_loader=train_loader,
        val_us_loader=val_us_loader,
        val_eu_loader=val_eu_loader,
        tokenizer=tokenizer,
    )

    # Resume from checkpoint if exists
    latest = config.checkpoints_dir / "latest_model.pt"
    if latest.exists():
        print(f"Resuming from {latest}")
        trainer.load_checkpoint(str(latest))

    # Train
    trainer.train()


if __name__ == "__main__":
    main()