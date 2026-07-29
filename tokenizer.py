"""Tokenizer utilities for GRAM legal LLM - BPE tokenizer using HuggingFace tokenizers."""

import json
import os
from pathlib import Path
from typing import List, Optional, Union

import torch
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from tokenizers.processors import TemplateProcessing

from config import config

# Optional HF token for gated datasets
HF_TOKEN = getattr(config, 'hf_token', None)


class LegalBPETokenizer:
    """BPE tokenizer trained on legal corpora with special tokens for jurisdictions."""

    SPECIAL_TOKENS = {
        "pad_token": "<|pad|>",
        "eos_token": "<|endoftext|>",
        "bos_token": "<|startoftext|>",
        "unk_token": "<|unk|>",
        "mask_token": "<|mask|>",
        "us_token": "<|us|>",
        "eu_token": "<|eu|>",
        "general_token": "<|general|>",
        "sep_token": "<|sep|>",
    }

    def __init__(
        self,
        vocab_size: int = None,
        min_frequency: int = 2,
        special_tokens: list = None,
        tokenizer_path: str = None,
    ):
        self.vocab_size = vocab_size or config.vocab_size
        self.min_frequency = min_frequency
        self.special_tokens = special_tokens or list(self.SPECIAL_TOKENS.values())
        self.tokenizer_path = Path(tokenizer_path) if tokenizer_path else config.tokenizer_dir / "tokenizer.json"
        self.tokenizer: Optional[Tokenizer] = None

        self._special_token_ids = {}

    def _create_tokenizer(self) -> Tokenizer:
        """Create a new BPE tokenizer with legal text optimizations."""
        tokenizer = Tokenizer(models.BPE(unk_token=self.SPECIAL_TOKENS["unk_token"]))

        tokenizer.normalizer = normalizers.Sequence([
            normalizers.NFD(),
            normalizers.Lowercase(),
            normalizers.StripAccents(),
        ])

        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.ByteLevel(add_prefix_space=False),
            pre_tokenizers.Digits(individual_digits=True),
        ])

        tokenizer.decoder = decoders.ByteLevel()

        return tokenizer

    def train(
        self,
        files: Union[str, List[str]],
        vocab_size: int = None,
        min_frequency: int = None,
        show_progress: bool = True,
    ) -> "LegalBPETokenizer":
        """Train BPE tokenizer on legal text files."""
        if isinstance(files, str):
            files = [files]

        tokenizer = self._create_tokenizer()

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size or self.vocab_size,
            min_frequency=min_frequency or self.min_frequency,
            special_tokens=self.special_tokens,
            show_progress=show_progress,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )

        tokenizer.train(files, trainer)

        tokenizer.post_processor = TemplateProcessing(
            single=f"{self.SPECIAL_TOKENS['bos_token']} $A {self.SPECIAL_TOKENS['eos_token']}",
            pair=f"{self.SPECIAL_TOKENS['bos_token']} $A {self.SPECIAL_TOKENS['sep_token']} $B {self.SPECIAL_TOKENS['eos_token']}",
            special_tokens=[
                (self.SPECIAL_TOKENS["bos_token"], self._get_token_id(tokenizer, self.SPECIAL_TOKENS["bos_token"])),
                (self.SPECIAL_TOKENS["eos_token"], self._get_token_id(tokenizer, self.SPECIAL_TOKENS["eos_token"])),
                (self.SPECIAL_TOKENS["sep_token"], self._get_token_id(tokenizer, self.SPECIAL_TOKENS["sep_token"])),
            ],
        )

        self.tokenizer = tokenizer
        self._cache_special_token_ids()
        return self

    def train_from_iterator(
        self,
        iterator,
        vocab_size: int = None,
        min_frequency: int = None,
        length: int = None,
        show_progress: bool = True,
    ) -> "LegalBPETokenizer":
        """Train BPE tokenizer from an iterator of strings."""
        tokenizer = self._create_tokenizer()

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size or self.vocab_size,
            min_frequency=min_frequency or self.min_frequency,
            special_tokens=self.special_tokens,
            show_progress=show_progress,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )

        tokenizer.train_from_iterator(iterator, trainer=trainer, length=length)

        tokenizer.post_processor = TemplateProcessing(
            single=f"{self.SPECIAL_TOKENS['bos_token']} $A {self.SPECIAL_TOKENS['eos_token']}",
            pair=f"{self.SPECIAL_TOKENS['bos_token']} $A {self.SPECIAL_TOKENS['sep_token']} $B {self.SPECIAL_TOKENS['eos_token']}",
            special_tokens=[
                (self.SPECIAL_TOKENS["bos_token"], self._get_token_id(tokenizer, self.SPECIAL_TOKENS["bos_token"])),
                (self.SPECIAL_TOKENS["eos_token"], self._get_token_id(tokenizer, self.SPECIAL_TOKENS["eos_token"])),
                (self.SPECIAL_TOKENS["sep_token"], self._get_token_id(tokenizer, self.SPECIAL_TOKENS["sep_token"])),
            ],
        )

        self.tokenizer = tokenizer
        self._cache_special_token_ids()
        return self

    def _get_token_id(self, tokenizer: Tokenizer, token: str) -> int:
        return tokenizer.token_to_id(token)

    def _cache_special_token_ids(self):
        """Cache special token IDs for fast access."""
        for name, token in self.SPECIAL_TOKENS.items():
            self._special_token_ids[name] = self.tokenizer.token_to_id(token)

    def save(self, path: str = None):
        """Save tokenizer to file."""
        save_path = Path(path) if path else self.tokenizer_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(str(save_path), pretty=True)

        # Also save special token map for easy loading
        special_map_path = save_path.parent / "special_tokens_map.json"
        with open(special_map_path, "w") as f:
            json.dump(self.SPECIAL_TOKENS, f, indent=2)

        print(f"Tokenizer saved to {save_path}")
        print(f"Vocab size: {self.tokenizer.get_vocab_size()}")

    @classmethod
    def load(cls, path: str = None, **kwargs) -> "LegalBPETokenizer":
        """Load tokenizer from file."""
        load_path = Path(path) if path else config.tokenizer_dir / "tokenizer.json"
        tokenizer = cls(**kwargs)
        tokenizer.tokenizer = Tokenizer.from_file(str(load_path))
        tokenizer._cache_special_token_ids()

        # Load special tokens map if exists
        special_map_path = load_path.parent / "special_tokens_map.json"
        if special_map_path.exists():
            with open(special_map_path) as f:
                tokenizer.SPECIAL_TOKENS = json.load(f)
                tokenizer._cache_special_token_ids()

        print(f"Tokenizer loaded from {load_path}")
        print(f"Vocab size: {tokenizer.tokenizer.get_vocab_size()}")
        return tokenizer

    def encode(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = True,
        return_tensors: str = None,
        max_length: int = None,
        padding: str = None,
        truncation: bool = True,
    ) -> Union[List[int], List[List[int]], torch.Tensor]:
        """Encode text to token IDs."""
        if isinstance(text, str):
            text = [text]
            single = True
        else:
            single = False

        encoded = self.tokenizer.encode_batch(text, add_special_tokens=add_special_tokens)

        if max_length is not None:
            for enc in encoded:
                if len(enc.ids) > max_length:
                    enc.ids = enc.ids[:max_length]
                    enc.type_ids = enc.type_ids[:max_length]
                    enc.attention_mask = enc.attention_mask[:max_length]

        if padding:
            max_len = max(len(e.ids) for e in encoded)
            for enc in encoded:
                pad_len = max_len - len(enc.ids)
                pad_id = self.pad_token_id
                enc.ids += [pad_id] * pad_len
                enc.attention_mask += [0] * pad_len
                enc.type_ids += [0] * pad_len

        if return_tensors == "pt":
            import torch
            input_ids = torch.tensor([e.ids for e in encoded], dtype=torch.long)
            attention_mask = torch.tensor([e.attention_mask for e in encoded], dtype=torch.long)
            if single:
                return {"input_ids": input_ids[0], "attention_mask": attention_mask[0]}
            return {"input_ids": input_ids, "attention_mask": attention_mask}

        if single:
            return encoded[0].ids
        return [e.ids for e in encoded]

    def decode(
        self,
        ids: Union[List[int], List[List[int]], torch.Tensor],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ) -> Union[str, List[str]]:
        """Decode token IDs to text."""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()

        if isinstance(ids[0], list):
            return self.tokenizer.decode_batch(ids, skip_special_tokens=skip_special_tokens)
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def get_vocab(self) -> dict:
        return self.tokenizer.get_vocab()

    def get_vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    @property
    def pad_token_id(self) -> int:
        return self._special_token_ids.get("pad_token", 0)

    @property
    def eos_token_id(self) -> int:
        return self._special_token_ids.get("eos_token", 1)

    @property
    def bos_token_id(self) -> int:
        return self._special_token_ids.get("bos_token", 2)

    @property
    def unk_token_id(self) -> int:
        return self._special_token_ids.get("unk_token", 3)

    @property
    def mask_token_id(self) -> int:
        return self._special_token_ids.get("mask_token", 4)

    @property
    def us_token_id(self) -> int:
        return self._special_token_ids.get("us_token", 5)

    @property
    def eu_token_id(self) -> int:
        return self._special_token_ids.get("eu_token", 6)

    @property
    def general_token_id(self) -> int:
        return self._special_token_ids.get("general_token", 7)

    @property
    def sep_token_id(self) -> int:
        return self._special_token_ids.get("sep_token", 8)

    def __len__(self) -> int:
        return self.get_vocab_size()

    def __repr__(self):
        return f"LegalBPETokenizer(vocab_size={self.get_vocab_size()}, special_tokens={list(self.SPECIAL_TOKENS.keys())})"


def get_tokenizer(path: str = None, **kwargs) -> LegalBPETokenizer:
    """Get or create tokenizer."""
    if path and Path(path).exists():
        return LegalBPETokenizer.load(path, **kwargs)
    return LegalBPETokenizer(**kwargs)


def train_tokenizer_from_datasets(
    us_data_path: str = None,
    eu_data_path: str = None,
    general_data_path: str = None,
    vocab_size: int = None,
    output_path: str = None,
    sample_size: int = 1000000,
) -> LegalBPETokenizer:
    """Train tokenizer on combined legal datasets."""
    import random
    from datasets import load_dataset as hf_load_dataset

    def sample_dataset(dataset, num_samples: int):
        if len(dataset) <= num_samples:
            return list(dataset["text"])
        indices = random.sample(range(len(dataset)), num_samples)
        return [dataset[i]["text"] for i in indices]

    texts = []

    if us_data_path and Path(us_data_path).exists():
        print(f"Loading US data from {us_data_path}")
        ds = hf_load_dataset("text", data_files=us_data_path, split="train")
        texts.extend(sample_dataset(ds, sample_size // 3))
    elif us_data_path:
        print(f"Loading US Caselaw Access Project from HF...")
        try:
            ds = hf_load_dataset("free-law/Caselaw_Access_Project", split="train", streaming=True, token=HF_TOKEN)
            texts.extend([ex["text"] for i, ex in zip(range(sample_size // 3), ds)])
        except Exception as e:
            print(f"Could not load Caselaw (gated): {e}")

    if eu_data_path and Path(eu_data_path).exists():
        print(f"Loading EU data from {eu_data_path}")
        ds = hf_load_dataset("text", data_files=eu_data_path, split="train")
        texts.extend(sample_dataset(ds, sample_size // 3))
    elif eu_data_path:
        print(f"Loading EUR-Lex from HF...")
        try:
            ds = hf_load_dataset("lex_glue", "eurlex", split="train", streaming=True, token=HF_TOKEN)
            texts.extend([ex["text"] for i, ex in zip(range(sample_size // 3), ds)])
        except:
            pass

    if general_data_path and Path(general_data_path).exists():
        print(f"Loading general data from {general_data_path}")
        ds = hf_load_dataset("text", data_files=general_data_path, split="train")
        texts.extend(sample_dataset(ds, sample_size // 3))

    if not texts:
        print("No data loaded, using sample text for demo...")
        texts = [
            "This is a sample legal text for tokenizer training. " * 100,
            "Another legal document with court cases and statutes. " * 100,
            "European Union regulation and directive text samples. " * 100,
        ] * 1000

    print(f"Training tokenizer on {len(texts)} documents...")
    tokenizer = LegalBPETokenizer(vocab_size=vocab_size or config.vocab_size)
    tokenizer.train_from_iterator(
        iter(texts),
        vocab_size=vocab_size or config.vocab_size,
        length=len(texts),
    )

    if output_path:
        tokenizer.save(output_path)
    else:
        tokenizer.save()

    return tokenizer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train legal BPE tokenizer")
    parser.add_argument("--vocab-size", type=int, default=config.vocab_size)
    parser.add_argument("--sample-size", type=int, default=1000000)
    parser.add_argument("--us-data", type=str, default=None)
    parser.add_argument("--eu-data", type=str, default=None)
    parser.add_argument("--general-data", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    train_tokenizer_from_datasets(
        us_data_path=args.us_data,
        eu_data_path=args.eu_data,
        general_data_path=args.general_data,
        vocab_size=args.vocab_size,
        output_path=args.output,
        sample_size=args.sample_size,
    )