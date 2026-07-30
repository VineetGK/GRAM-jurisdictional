# GRAM for Law: Jurisdiction-Aware Legal Language Model.

Implementation of **GRAM (Gradient-Routed Auxiliary Modules)** for modular pretraining of a legal LLM with US and EU jurisdiction modules, plus a Streamlit chat interface.

Based on: *Modular Pretraining Enables Access Control* (GRAM paper)

## Architecture

```
┌─────────────────────────────────────────────┐
│           GRAM Transformer Block            │
├─────────────────────────────────────────────┤
│  Core MLP (always active)                   │
│  ┌─────────────┐  ┌─────────────┐           │
│  │ US Module   │  │ EU Module   │  (conditionally active) │
│  │ MLP + LN    │  │ MLP + LN    │           │
│  └─────────────┘  └─────────────┘           │
└─────────────────────────────────────────────┘
```

### Three Configurations
| Config | Core | US Module | EU Module |
|--------|------|-----------|-----------|
| **Full** | ✓ | ✓ | ✓ |
| **US-only** | ✓ | ✓ | ✗ |
| **EU-only** | ✓ | ✗ | ✓ |

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Train Tokenizer
```bash
python tokenizer.py --vocab-size 32000
```

### 2. Train Model (GRAM)
```bash
python train.py
```

Training has two phases:
- **Phase 1 (Warmup)**: General corpus, updates all parameters (~2000 steps)
- **Phase 2 (GRAM)**: Alternates US/EU batches with gradient routing (~8000 steps)

### 3. Run Evaluation (Ablation)
```bash
python evaluate.py
```

### 4. Launch Streamlit App
```bash
streamlit run app.py
```

## Files

| File | Description |
|------|-------------|
| `config.py` | Hyperparameters, paths, model config |
| `tokenizer.py` | BPE tokenizer training/loading |
| `datasets.py` | Data loading, chunking, DataLoaders |
| `model.py` | GRAM Transformer with modular MLPs |
| `train.py` | Training loop with GRAM gradient routing |
| `evaluate.py` | Ablation experiments (Full/US-only/EU-only) |
| `app.py` | Streamlit chat interface |

## GRAM Gradient Routing

During **Phase 2 (GRAM phase)**:

| Batch Jurisdiction | Core Params | US Module | EU Module |
|-------------------|-------------|-----------|-----------|
| US | Frozen* | **Updated** | Frozen |
| EU | Frozen* | Frozen | **Updated** |
| General | Updated | Updated | Updated |

*Configurable via `freeze_core_during_gram` and `freeze_other_module_during_gram`

## Model Details

- **Architecture**: 12-layer decoder-only Transformer (~100M params)
- **Hidden size**: 768, **Heads**: 12, **FFN**: 3072
- **Modules**: Core MLP (3072) + US MLP (768) + EU MLP (768) per layer
- **Context**: 512 tokens
- **Tokenizer**: BPE, 32k vocab, legal special tokens (`<|us|>`, `<|eu|>`, `<|general|>`)

## Datasets

| Dataset | Source | Jurisdiction |
|---------|--------|--------------|
| US Caselaw | HuggingFace `free-law/Caselaw_Access_Project` | US |
| EU Law | HuggingFace `lex_glue` (EUR-Lex) | EU |
| General | Wikipedia (optional) | General |

## Streamlit App Features

- **Mode selector**: Full / US-only / EU-only
- **Chat interface** with conversation history
- **Config badges** showing active mode on each response
- **Generation controls**: temperature, top-k, top-p, repetition penalty
- **Example prompts** for US, EU, and comparative questions

## Training Monitoring

```bash
tensorboard --logdir logs/
```

## Customization

Edit `config.py` to adjust:
- Model size (`n_layers`, `n_heads`, `n_embd`)
- Training steps, batch size, learning rate
- GRAM routing behavior
- Data paths and sample sizes

## TODO (Production)

- [ ] Distributed training (DDP/FSDP)
- [ ] Flash Attention 2
- [ ] Gradient checkpointing
- [ ] Better data cleaning for legal texts
- [ ] Instruction tuning / RLHF
- [ ] Quantization (GGUF/GPTQ) for deployment
- [ ] RAG integration for citations

## License

Research prototype. Legal datasets have their own licenses (public domain for US caselaw, EUR-Lex for EU law).
