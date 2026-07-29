"""
Evaluation & Ablation Module for GRAM Legal LLM
================================================

Provides comprehensive evaluation including:
- Perplexity evaluation on US/EU/General test sets
- Jurisdiction-specific legal benchmarks
- Ablation studies (core-only, core+US, core+EU, core+General, full)
- Generation quality metrics
- Parameter efficiency analysis
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import config
from model import GRAMModel, ModelConfig
from tokenizer import load_tokenizer, encode_with_jurisdiction, SPECIAL_TOKENS
from datasets import (
    download_us_dataset, download_eu_dataset, download_general_dataset,
    LegalDataset, collate_fn, JurisdictionSample
)


@dataclass
class EvalResult:
    jurisdiction: str
    perplexity: float
    loss: float
    num_tokens: int
    num_samples: int
    metrics: Dict[str, float]


@dataclass
class AblationResult:
    config_name: str
    active_jurisdictions: List[str]
    core_params: int
    adapter_params: int
    total_params: int
    results: Dict[str, EvalResult]
    avg_perplexity: float


class Evaluator:
    """Evaluator for GRAM model with ablation support."""
    
    def __init__(
        self,
        model: GRAMModel,
        tokenizer,
        device: str = "cuda",
        max_seq_len: int = 1024,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_seq_len = max_seq_len
        
        self.model.to(device)
        self.model.eval()
    
    @torch.no_grad()
    def evaluate_perplexity(
        self,
        samples: List[JurisdictionSample],
        jurisdiction: str,
        batch_size: int = 8,
        max_samples: Optional[int] = None,
    ) -> EvalResult:
        """Evaluate perplexity on a set of samples."""
        
        if max_samples:
            samples = samples[:max_samples]
        
        dataset = LegalDataset(
            samples,
            self.tokenizer,
            chunk_size=512,
            max_seq_len=self.max_seq_len,
            shuffle=False,
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
        )
        
        total_loss = 0.0
        total_tokens = 0
        num_samples = 0
        
        for batch in tqdm(dataloader, desc=f"Evaluating {jurisdiction}"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            active_jurisdiction = jurisdiction if jurisdiction != "general" else "core"
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                jurisdiction=active_jurisdiction,
            )
            
            loss = outputs["loss"]
            if loss is not None:
                num_tokens = (labels != -100).sum().item()
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
                num_samples += input_ids.size(0)
        
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')
        
        return EvalResult(
            jurisdiction=jurisdiction,
            perplexity=perplexity,
            loss=avg_loss,
            num_tokens=total_tokens,
            num_samples=num_samples,
            metrics={},
        )
    
    @torch.no_grad()
    def evaluate_generation(
        self,
        prompts: List[str],
        jurisdiction: str,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> List[Dict[str, Any]]:
        """Generate completions for prompts."""
        
        active_jurisdiction = jurisdiction if jurisdiction != "general" else "core"
        self.model.set_jurisdiction(active_jurisdiction)
        
        results = []
        for prompt in prompts:
            encoded = encode_with_jurisdiction(
                self.tokenizer, prompt, jurisdiction, self.max_seq_len // 2
            )
            input_ids = torch.tensor([encoded["input_ids"]], device=self.device)
            attention_mask = torch.tensor([encoded["attention_mask"]], device=self.device)
            
            generated = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            
            generated_text = self.tokenizer.decode(
                generated[0][len(encoded["input_ids"]):],
                skip_special_tokens=True,
            )
            
            results.append({
                "prompt": prompt,
                "completion": generated_text,
                "jurisdiction": jurisdiction,
            })
        
        return results


def load_test_samples(
    data_dir: Path, 
    max_per_jurisdiction: int = 1000
) -> Dict[str, List[JurisdictionSample]]:
    """Load test samples for each jurisdiction (90/10 train/test split)."""
    
    samples = {}
    
    us_samples = download_us_dataset(data_dir, max_per_jurisdiction)
    split_idx = int(len(us_samples) * 0.9)
    samples["US"] = us_samples[split_idx:]
    
    eu_samples = download_eu_dataset(data_dir, max_per_jurisdiction)
    split_idx = int(len(eu_samples) * 0.9)
    samples["EU"] = eu_samples[split_idx:]
    
    gen_samples = download_general_dataset(data_dir, max_per_jurisdiction)
    split_idx = int(len(gen_samples) * 0.9)
    samples["general"] = gen_samples[split_idx:]
    
    return samples


def run_ablation_study(
    model: GRAMModel,
    tokenizer,
    test_samples: Dict[str, List[JurisdictionSample]],
    device: str = "cuda",
) -> List[AblationResult]:
    """Run comprehensive ablation study."""
    
    evaluator = Evaluator(model, tokenizer, device)
    
    ablations = [
        ("core_only", ["core"]),
        ("core_us", ["core", "US"]),
        ("core_eu", ["core", "EU"]),
        ("core_general", ["core", "general"]),
        ("core_us_eu", ["core", "US", "EU"]),
        ("core_all", ["core", "US", "EU", "general"]),
    ]
    
    results = []
    
    for config_name, active_jurs in ablations:
        print(f"\n{'='*60}")
        print(f"Ablation: {config_name} - Active: {active_jurs}")
        print(f"{'='*60}")
        
        jurisdiction_results = {}
        perplexities = []
        
        for jur in ["US", "EU", "general"]:
            if jur == "general":
                active_jur = "core" if "general" not in active_jurs else "general"
            else:
                active_jur = jur if jur in active_jurs else "core"
            
            result = evaluator.evaluate_perplexity(
                test_samples[jur],
                active_jur,
                batch_size=8,
                max_samples=config.eval_samples_us if jur == "US" else config.eval_samples_eu,
            )
            
            jurisdiction_results[jur] = result
            perplexities.append(result.perplexity)
            
            print(f"  {jur}: PPL={result.perplexity:.2f}, Loss={result.loss:.4f}, Tokens={result.num_tokens}")
        
        core_params = model.get_jurisdiction_parameters("core")
        adapter_params = sum(
            model.get_jurisdiction_parameters(jur) - core_params
            for jur in ["US", "EU", "general"]
            if jur in active_jurs
        )
        
        ablation_result = AblationResult(
            config_name=config_name,
            active_jurisdictions=active_jurs,
            core_params=core_params,
            adapter_params=adapter_params,
            total_params=core_params + adapter_params,
            results=jurisdiction_results,
            avg_perplexity=sum(perplexities) / len(perplexities),
        )
        results.append(ablation_result)
        
        print(f"  Average PPL: {ablation_result.avg_perplexity:.2f}")
        print(f"  Params: Core={core_params/1e6:.2f}M, Adapters={adapter_params/1e6:.2f}M, Total={(core_params+adapter_params)/1e6:.2f}M")
    
    return results


def evaluate_checkpoint(
    checkpoint_path: str,
    test_samples: Dict[str, List[JurisdictionSample]],
    model_config: ModelConfig,
    tokenizer,
    device: str = "cuda",
    run_ablation: bool = True,
) -> Dict[str, Any]:
    """Evaluate a single checkpoint."""
    
    model = GRAMModel(model_config)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict.get("model_state_dict", state_dict))
    model.to(device)
    model.eval()
    
    evaluator = Evaluator(model, tokenizer, device)
    
    results = {}
    
    for jur in ["US", "EU", "general"]:
        result = evaluator.evaluate_perplexity(
            test_samples[jur],
            jur if jur != "general" else "core",
            batch_size=8,
            max_samples=config.eval_samples_us if jur == "US" else config.eval_samples_eu,
        )
        results[jur] = asdict(result)
        print(f"{jur}: PPL={result.perplexity:.2f}, Loss={result.loss:.4f}")
    
    if run_ablation:
        ablation_results = run_ablation_study(model, tokenizer, test_samples, device)
        results["ablations"] = [asdict(r) for r in ablation_results]
    
    return results


def compare_checkpoints(
    checkpoint_paths: List[str],
    test_samples: Dict[str, List[JurisdictionSample]],
    model_config: ModelConfig,
    tokenizer,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Compare multiple checkpoints."""
    
    all_results = {}
    
    for ckpt_path in checkpoint_paths:
        print(f"\nEvaluating {ckpt_path}")
        results = evaluate_checkpoint(
            ckpt_path, test_samples, model_config, tokenizer, device, run_ablation=False
        )
        all_results[ckpt_path] = results
    
    return all_results


def generate_samples(
    model: GRAMModel,
    tokenizer,
    prompts: Dict[str, List[str]],
    device: str = "cuda",
    max_new_tokens: int = 256,
) -> Dict[str, List[Dict]]:
    """Generate samples for qualitative evaluation."""
    
    evaluator = Evaluator(model, tokenizer, device)
    
    all_generations = {}
    
    for jur, jur_prompts in prompts.items():
        print(f"\nGenerating for {jur}...")
        generations = evaluator.evaluate_generation(
            jur_prompts,
            jur,
            max_new_tokens=max_new_tokens,
        )
        all_generations[jur] = generations
        
        for gen in generations[:3]:
            print(f"  Prompt: {gen['prompt'][:100]}...")
            print(f"  Completion: {gen['completion'][:200]}...")
            print()
    
    return all_generations


def compute_parameter_efficiency(results: List[AblationResult]) -> Dict[str, float]:
    """Compute parameter efficiency metrics."""
    
    core_only = next(r for r in results if r.config_name == "core_only")
    core_all = next(r for r in results if r.config_name == "core_all")
    
    param_increase = (core_all.total_params - core_only.total_params) / core_only.total_params * 100
    ppl_improvement = (core_only.avg_perplexity - core_all.avg_perplexity) / core_only.avg_perplexity * 100
    
    return {
        "parameter_increase_pct": param_increase,
        "perplexity_improvement_pct": ppl_improvement,
        "params_per_ppl_point": (
            (core_all.total_params - core_only.total_params) / 
            (core_only.avg_perplexity - core_all.avg_perplexity)
        ) if core_only.avg_perplexity != core_all.avg_perplexity else float('inf'),
    }


def save_results(results: Dict[str, Any], output_path: Path):
    """Save evaluation results to JSON."""
    
    def serialize(obj):
        if isinstance(obj, (EvalResult, AblationResult)):
            return asdict(obj)
        elif isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, (set, tuple)):
            return list(obj)
        return obj
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=serialize)
    
    print(f"Results saved to {output_path}")


def load_results(input_path: Path) -> Dict[str, Any]:
    """Load evaluation results from JSON."""
    with open(input_path, "r") as f:
        return json.load(f)


def main():
    print("=" * 60)
    print("GRAM Legal LLM Evaluation")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    tokenizer = load_tokenizer(str(config.tokenizer_dir))
    
    model_config = ModelConfig(
        vocab_size=config.tokenizer_vocab_size,
        max_seq_len=config.max_seq_len,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        dropout=config.dropout,
        adapter_rank=config.adapter_rank,
        adapter_alpha=config.adapter_alpha,
    )
    
    print("Loading test samples...")
    test_samples = load_test_samples(config.data_dir, config.eval_samples_us + config.eval_samples_eu)
    
    checkpoints = list(config.checkpoint_dir.glob("checkpoint_step_*.pt"))
    if not checkpoints:
        print("No checkpoints found!")
        return
    
    latest_ckpt = max(checkpoints, key=lambda x: int(x.stem.split("_")[-1]))
    print(f"Evaluating latest checkpoint: {latest_ckpt}")
    
    results = evaluate_checkpoint(
        str(latest_ckpt),
        test_samples,
        model_config,
        tokenizer,
        device,
        run_ablation=True,
    )
    
    output_path = config.output_dir / f"eval_results_{latest_ckpt.stem}.json"
    save_results(results, output_path)
    
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    for jur in ["US", "EU", "general"]:
        r = results[jur]
        print(f"{jur}: PPL={r['perplexity']:.2f}, Loss={r['loss']:.4f}")
    
    if "ablations" in results:
        print("\nAblation Results:")
        for ablation in results["ablations"]:
            print(f"  {ablation['config_name']}: avg PPL={ablation['avg_perplexity']:.2f}, "
                  f"params={ablation['total_params']/1e6:.2f}M")


if __name__ == "__main__":
    main()