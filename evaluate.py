"""Evaluation script for GRAM legal LLM with ablation experiments."""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tabulate import tabulate

from config import config
from model import GRAMTransformer, ModelConfig, create_model
from tokenizer import LegalBPETokenizer, get_tokenizer
from data import (
    LegalTextDataset,
    LegalIterableDataset,
    collate_batch,
    create_dummy_loader,
    JurisdictionBatch,
)


class GRAMEvaluator:
    """Evaluator for GRAM model with ablation experiments."""

    def __init__(
        self,
        model: GRAMTransformer,
        tokenizer: LegalBPETokenizer,
        cfg: config = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg or config

        self.device = torch.device(self.cfg.device)
        self.dtype = self.cfg.get_dtype()
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def evaluate_dataset(
        self,
        loader: DataLoader,
        name: str,
        enable_us: bool = True,
        enable_eu: bool = True,
    ) -> Dict[str, float]:
        """Evaluate on a single dataset."""
        total_loss = 0.0
        total_tokens = 0
        total_correct = 0

        for batch in loader:
            input_ids = batch.input_ids.to(self.device)
            attention_mask = batch.attention_mask.to(self.device)
            labels = batch.labels.to(self.device)

            with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    enable_us_module=enable_us,
                    enable_eu_module=enable_eu,
                )
                loss = outputs["loss"]
                logits = outputs["logits"]

            # Accumulate loss
            total_loss += loss.item() * input_ids.numel()
            total_tokens += input_ids.numel()

            # Accuracy (next token prediction)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            preds = shift_logits.argmax(dim=-1)
            mask = shift_labels != -100
            total_correct += (preds == shift_labels).masked_select(mask).sum().item()

        avg_loss = total_loss / total_tokens
        ppl = math.exp(min(avg_loss, 20))
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0

        return {
            f"{name}_loss": avg_loss,
            f"{name}_ppl": ppl,
            f"{name}_acc": accuracy,
        }

    def run_ablation(
        self,
        us_loader: DataLoader,
        eu_loader: DataLoader,
        general_loader: Optional[DataLoader] = None,
    ) -> List[Dict]:
        """
        Run ablation experiments across three configurations:
        1. Full (core + US + EU)
        2. US-only (core + US, EU disabled)
        3. EU-only (core + EU, US disabled)
        """
        configs = [
            {"name": "Full", "enable_us": True, "enable_eu": True},
            {"name": "US-only", "enable_us": True, "enable_eu": False},
            {"name": "EU-only", "enable_us": False, "enable_eu": True},
        ]

        results = []

        for cfg in configs:
            print(f"\nEvaluating {cfg['name']} config...")
            print(f"  US module: {'ON' if cfg['enable_us'] else 'OFF'}, EU module: {'ON' if cfg['enable_eu'] else 'OFF'}")

            config_results = {"config": cfg["name"]}

            # Evaluate on US test data
            us_metrics = self.evaluate_dataset(
                us_loader, "us",
                enable_us=cfg["enable_us"],
                enable_eu=cfg["enable_eu"],
            )
            config_results.update(us_metrics)
            print(f"  US Test: loss={us_metrics['us_loss']:.4f}, ppl={us_metrics['us_ppl']:.2f}, acc={us_metrics['us_acc']:.4f}")

            # Evaluate on EU test data
            eu_metrics = self.evaluate_dataset(
                eu_loader, "eu",
                enable_us=cfg["enable_us"],
                enable_eu=cfg["enable_eu"],
            )
            config_results.update(eu_metrics)
            print(f"  EU Test: loss={eu_metrics['eu_loss']:.4f}, ppl={eu_metrics['eu_ppl']:.2f}, acc={eu_metrics['eu_acc']:.4f}")

            # Evaluate on general if provided
            if general_loader:
                gen_metrics = self.evaluate_dataset(
                    general_loader, "general",
                    enable_us=cfg["enable_us"],
                    enable_eu=cfg["enable_eu"],
                )
                config_results.update(gen_metrics)
                print(f"  General Test: loss={gen_metrics['general_loss']:.4f}, ppl={gen_metrics['general_ppl']:.2f}")

            results.append(config_results)

        return results

    def print_ablation_table(self, results: List[Dict]):
        """Print ablation results in a formatted table."""
        headers = ["Config", "US Loss", "US PPL", "US Acc", "EU Loss", "EU PPL", "EU Acc"]
        rows = []

        for r in results:
            row = [
                r["config"],
                f"{r['us_loss']:.4f}",
                f"{r['us_ppl']:.2f}",
                f"{r['us_acc']:.4f}",
                f"{r['eu_loss']:.4f}",
                f"{r['eu_ppl']:.2f}",
                f"{r['eu_acc']:.4f}",
            ]
            rows.append(row)

        print("\n" + "=" * 80)
        print("ABLATION RESULTS")
        print("=" * 80)
        print(tabulate(rows, headers=headers, tablefmt="grid"))

        # Compute deltas relative to Full
        full = next(r for r in results if r["config"] == "Full")
        print("\nPerformance Deltas (relative to Full):")
        delta_headers = ["Config", "Delta US Loss", "Delta US PPL", "Delta EU Loss", "Delta EU PPL"]
        delta_rows = []

        for r in results:
            if r["config"] == "Full":
                continue
            delta_row = [
                r["config"],
                f"{r['us_loss'] - full['us_loss']:+.4f}",
                f"{r['us_ppl'] - full['us_ppl']:+.2f}",
                f"{r['eu_loss'] - full['eu_loss']:+.4f}",
                f"{r['eu_ppl'] - full['eu_ppl']:+.2f}",
            ]
            delta_rows.append(delta_row)

        print(tabulate(delta_rows, headers=delta_headers, tablefmt="grid"))

    def save_results(self, results: List[Dict], path: str):
        """Save evaluation results to JSON."""
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {path}")


def load_test_data(
    tokenizer: LegalBPETokenizer,
    jurisdiction: str,
    sample_size: int,
    max_length: int = 512,
) -> DataLoader:
    """Load test data for a jurisdiction."""
    try:
        if jurisdiction == "US":
            ds = LegalIterableDataset(
                "free-law/Caselaw_Access_Project", "train",
                tokenizer, "US",
                max_length=max_length,
                num_samples=sample_size,
            )
        elif jurisdiction == "EU":
            ds = LegalIterableDataset(
                "lex_glue", "train",
                tokenizer, "EU",
                max_length=max_length,
                num_samples=sample_size,
            )
        elif jurisdiction == "general":
            ds = LegalIterableDataset(
                "wikipedia", "train",
                tokenizer, "general",
                max_length=max_length,
                num_samples=sample_size,
            )
        else:
            raise ValueError(f"Unknown jurisdiction: {jurisdiction}")

        return DataLoader(
            ds,
            batch_size=config.eval_batch_size,
            num_workers=0,
            collate_fn=lambda b: collate_batch(b, tokenizer.pad_token_id, max_length),
        )
    except Exception as e:
        print(f"Could not load {jurisdiction} test data: {e}, using dummy data")
        return create_dummy_loader(tokenizer, jurisdiction, config.eval_batch_size, sample_size, max_length)


def main():
    """Main evaluation entry point."""
    print("=" * 60)
    print("GRAM Legal LLM Evaluation - Ablation Experiments")
    print("=" * 60)

    # Load tokenizer
    tokenizer_path = config.tokenizer_dir / "tokenizer.json"
    if not tokenizer_path.exists():
        print(f"Tokenizer not found at {tokenizer_path}")
        print("Run train.py first or train a tokenizer")
        return

    tokenizer = LegalBPETokenizer.load(str(tokenizer_path))
    print(f"Tokenizer vocab size: {len(tokenizer)}")

    # Load model
    model_path = config.checkpoints_dir / "best_model.pt"
    if not model_path.exists():
        model_path = config.checkpoints_dir / "final_model.pt"
    if not model_path.exists():
        # Find latest checkpoint
        checkpoints = sorted(config.checkpoints_dir.glob("checkpoint_step_*.pt"))
        if checkpoints:
            model_path = checkpoints[-1]
        else:
            print(f"No model checkpoint found in {config.checkpoints_dir}")
            return

    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu")

    model_cfg = ModelConfig(
        vocab_size=len(tokenizer),
        enable_us_module=True,
        enable_eu_module=True,
    )
    model = create_model(model_cfg)
    model.load_state_dict(checkpoint["model"])
    print(f"Model loaded (step {checkpoint.get('step', 'unknown')})")
    print(f"Parameters: {model.num_parameters():,}")

    # Create evaluator
    evaluator = GRAMEvaluator(model, tokenizer)

    # Load test data
    print("\nLoading test data...")
    us_loader = load_test_data(tokenizer, "US", config.eval_us_sample_size)
    eu_loader = load_test_data(tokenizer, "EU", config.eval_eu_sample_size)

    try:
        general_loader = load_test_data(tokenizer, "general", config.eval_us_sample_size // 2)
    except:
        general_loader = None

    # Run ablation
    print("\nRunning ablation experiments...")
    results = evaluator.run_ablation(us_loader, eu_loader, general_loader)

    # Print results
    evaluator.print_ablation_table(results)

    # Save results
    output_path = config.logs_dir / "ablation_results.json"
    evaluator.save_results(results, str(output_path))

    # Also test generation for each config
    print("\n" + "=" * 80)
    print("GENERATION EXAMPLES")
    print("=" * 80)

    test_prompts = [
        "The court held that",
        "Article 101 of the Treaty",
        "In accordance with the statute",
    ]

    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        input_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")["input_ids"].unsqueeze(0).to(evaluator.device)

        for cfg_name in ["Full", "US-only", "EU-only"]:
            cfg = next(c for c in config.eval_configs if c["name"] == cfg_name)
            generated = model.generate(
                input_ids,
                max_new_tokens=50,
                temperature=0.8,
                top_k=50,
                top_p=0.95,
                enable_us_module=cfg["enable_us_module"],
                enable_eu_module=cfg["enable_eu_module"],
                eos_token_id=tokenizer.eos_token_id,
            )
            text = tokenizer.decode(generated[0], skip_special_tokens=True)
            print(f"  [{cfg_name}] {text}")


if __name__ == "__main__":
    main()