"""
Tokenizer module for GRAM Legal LLM
====================================

Builds/loads a BPE tokenizer trained on combined US + EU + General corpora.
Uses HuggingFace tokenizers (byte-level BPE) for fast training/encoding.

Install: pip install tokenizers transformers
"""

import os
import json
from pathlib import Path
from typing import List, Optional, Iterator
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from tokenizers.normalizers import NFKC, Sequence, Lowercase
from transformers import PreTrainedTokenizerFast
from config import config


SPECIAL_TOKENS = {
    "pad_token": "<|pad|>",
    "unk_token": "<|unk|>",
    "bos_token": "<|bos|>",
    "eos_token": "<|eos|>",
    "us_token": "<|us|>",
    "eu_token": "<|eu|>",
    "general_token": "<|general|>",
}

SPECIAL_TOKENS_LIST = list(SPECIAL_TOKENS.values())


def get_training_corpus(data_dir: str, max_samples: int = None) -> Iterator[str]:
    """Yield text chunks for tokenizer training from raw data files."""
    data_path = Path(data_dir)
    for file_path in data_path.glob("*.txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            chunks = [text[i:i+10000] for i in range(0, len(text), 10000)]
            for chunk in chunks:
                yield chunk
                if max_samples and max_samples <= 0:
                    return
                if max_samples:
                    max_samples -= 1


def train_tokenizer(
    data_dir: str,
    vocab_size: int = 32000,
    min_frequency: int = 2,
    output_dir: str = "./tokenizer",
    max_samples: int = None,
) -> Tokenizer:
    """Train a byte-level BPE tokenizer on the combined corpus."""
    print(f"Training BPE tokenizer on data from {data_dir}...")
    
    tokenizer = Tokenizer(models.BPE(unk_token=SPECIAL_TOKENS["unk_token"]))
    
    tokenizer.normalizer = Sequence([NFKC(), Lowercase()])
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()
    
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"{SPECIAL_TOKENS['bos_token']} $A {SPECIAL_TOKENS['eos_token']}",
        pair=f"{SPECIAL_TOKENS['bos_token']} $A {SPECIAL_TOKENS['eos_token']} $B {SPECIAL_TOKENS['eos_token']}",
        special_tokens=[
            (SPECIAL_TOKENS["bos_token"], 1),
            (SPECIAL_TOKENS["eos_token"], 2),
        ],
    )
    
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS_LIST,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    
    corpus = get_training_corpus(data_dir, max_samples)
    tokenizer.train_from_iterator(corpus, trainer=trainer)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path / "tokenizer.json"))
    
    config = {
        "vocab_size": vocab_size,
        "max_seq_len": config.max_seq_len,
        "special_tokens": SPECIAL_TOKENS,
    }
    with open(output_path / "tokenizer_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Tokenizer saved to {output_path}")
    return tokenizer


def load_tokenizer(tokenizer_dir: str) -> PreTrainedTokenizerFast:
    """Load trained tokenizer as HuggingFace PreTrainedTokenizerFast."""
    tokenizer_path = Path(tokenizer_dir) / "tokenizer.json"
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}. Run train_tokenizer() first.")
    
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_path),
        pad_token=SPECIAL_TOKENS["pad_token"],
        unk_token=SPECIAL_TOKENS["unk_token"],
        bos_token=SPECIAL_TOKENS["bos_token"],
        eos_token=SPECIAL_TOKENS["eos_token"],
        additional_special_tokens=[
            SPECIAL_TOKENS["us_token"],
            SPECIAL_TOKENS["eu_token"],
            SPECIAL_TOKENS["general_token"],
        ],
        model_max_length=config.max_seq_len,
        padding_side="right",
        truncation_side="right",
    )
    return tokenizer


def get_jurisdiction_token(tokenizer: PreTrainedTokenizerFast, jurisdiction: str) -> int:
    """Get token ID for jurisdiction marker."""
    token_map = {
        "US": SPECIAL_TOKENS["us_token"],
        "EU": SPECIAL_TOKENS["eu_token"],
        "general": SPECIAL_TOKENS["general_token"],
    }
    token = token_map.get(jurisdiction, SPECIAL_TOKENS["general_token"])
    return tokenizer.convert_tokens_to_ids(token)


def encode_with_jurisdiction(
    tokenizer: PreTrainedTokenizerFast,
    text: str,
    jurisdiction: str,
    max_length: int = None,
    add_special_tokens: bool = True,
) -> dict:
    """Encode text with jurisdiction token prepended."""
    if max_length is None:
        max_length = config.max_seq_len
    
    jurisdiction_token = get_jurisdiction_token(tokenizer, jurisdiction)
    
    encoding = tokenizer(
        text,
        max_length=max_length - 2,
        truncation=True,
        padding=False,
        add_special_tokens=False,
        return_tensors="pt",
    )
    
    input_ids = encoding["input_ids"][0].tolist()
    input_ids = [tokenizer.bos_token_id, jurisdiction_token] + input_ids + [tokenizer.eos_token_id]
    
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
    
    attention_mask = [1] * len(input_ids)
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "jurisdiction": jurisdiction,
    }


def decode_tokens(tokenizer: PreTrainedTokenizerFast, token_ids: list, skip_special_tokens: bool = True) -> str:
    """Decode token IDs to text."""
    return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)


def save_tokenizer_config(tokenizer_dir: str, config: dict):
    """Save tokenizer configuration."""
    with open(Path(tokenizer_dir) / "tokenizer_config.json", "w") as f:
        json.dump(config, f, indent=2)


def load_tokenizer_config(tokenizer_dir: str) -> dict:
    """Load tokenizer configuration."""
    config_path = Path(tokenizer_dir) / "tokenizer_config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            return json.load(f)
    return {}


if __name__ == "__main__":
    from tokenizers import pre_tokenizers, decoders, processors
    
    train_tokenizer(
        data_dir=str(config.data_dir),
        vocab_size=config.tokenizer_vocab_size,
        min_frequency=config.tokenizer_min_frequency,
        output_dir=str(config.tokenizer_dir),
        max_samples=config.max_samples_us + config.max_samples_eu + config.max_samples_general,
    )
    
    tokenizer = load_tokenizer(str(config.tokenizer_dir))
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Special tokens: {SPECIAL_TOKENS}")
    
    test_text = "This is a test legal document about contract law."
    encoded = encode_with_jurisdiction(tokenizer, test_text, "US")
    print(f"Encoded: {encoded}")
    decoded = decode_tokens(tokenizer, encoded["input_ids"])
    print(f"Decoded: {decoded}")