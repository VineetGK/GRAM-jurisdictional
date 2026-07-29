# GRAM for Law: Generic + Region-Adaptive Module Legal LLM

A modular legal language model with jurisdiction-specific adapters (US, EU, General) built on a shared Transformer core using LoRA.

## Architecture

```
Input → Embedding → Core Transformer (12 layers) → [Core + Jurisdiction Adapter] → LM Head → Logits
                                                     ├── US Adapter (LoRA rank=16)
                                                     ├── EU Adapter (LoRA rank=16)
                                                     └── General Adapter (LoRA rank=16)
```

**Key Features:**
- **Gradient Routing**: During training, gradients flow only to core + active jurisdiction adapter
- **Parameter Efficient**: ~768K adapter params per jurisdiction vs 124M core params
- **Modular Inference**: Switch jurisdictions at runtime without reloading
- **Ablation Ready**: Evaluate core-only, core+US, core+EU, core+general configs

## Files

| File | Description |
|------|-------------|
| `config.py` | Hyperparameters, paths, device config |
| `tokenizer.py` | BPE tokenizer training/loading (32K vocab) |
| `datasets.py` | US/EU/General dataset download, preprocessing, DataLoaders |
| `model.py` | GRAMModel with LoRA adapters per jurisdiction |
| `train.py` | GRAM training loop with gradient routing |
| `evaluate.py` | Perplexity evaluation + ablation study |
| `app.py` | Streamlit chat interface with jurisdiction selector |

## Quick Start

```bash
cd "E:\Projects Github\GRAM for Law"

# Create environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install torch transformers tokenizers datasets accelerate streamlit tqdm

# 1. Train tokenizer (downloads data, trains 32K BPE)
python tokenizer.py

# 2. Train model (GRAM gradient routing)
python train.py

# 3. Evaluate (perplexity + ablations)
python evaluate.py

# 4. Chat interface
streamlit run app.py
```

## Configuration

Edit `config.py` to customize:
- Model size: `d_model`, `n_layers`, `n_heads`
- Adapter config: `adapter_rank`, `adapter_alpha`
- Training: `batch_size`, `learning_rate`, `max_steps`
- Data: `max_samples_us`, `max_samples_eu`, `max_samples_general`
- Paths: `data_dir`, `tokenizer_dir`, `checkpoint_dir`

## Training Details

**Gradient Routing Modes:**
- `jurisdiction` (default): Core + active jurisdiction adapter
- `all`: All parameters
- `core_only`: Freeze all adapters

**Jurisdiction Schedules:**
- `mixed`: Random batch jurisdiction
- `sequential`: US epoch → EU epoch → General epoch

## Data Sources

| Jurisdiction | Dataset | Source |
|-------------|---------|--------|
| US | Caselaw Access Project | Harvard Law / Free Law Project |
| EU | EurLex | European Union Law |
| General | Wikipedia | Wikimedia |

*Downloads automatically on first run. Uses dummy data if unavailable.*

## Evaluation

```bash
python evaluate.py
```

Outputs:
- Perplexity per jurisdiction (US, EU, General)
- Ablation table: Full / US-only / EU-only / Core-only
- Parameter counts per configuration

## Streamlit App

```bash
streamlit run app.py
```

Features:
- Sidebar: Checkpoint selector, jurisdiction (US/EU/General), generation params
- Chat interface with history
- Model architecture details panel
- Special tokens reference

## Requirements

```
torch>=2.0.0
transformers>=4.30.0
tokenizers>=0.13.0
datasets>=2.12.0
accelerate>=0.20.0
streamlit>=1.25.0
tqdm>=4.65.0
numpy>=1.24.0
```

## Project Structure

```
GRAM for Law/
├── config.py
├── tokenizer.py
├── datasets.py
├── model.py
├── train.py
├── evaluate.py
├── app.py
├── requirements.txt
├── README.md
├── data/              # Downloaded corpora
├── tokenizer/         # Trained BPE tokenizer
├── checkpoints/       # Model checkpoints
├── logs/              # Training logs
└── outputs/           # Evaluation results
```