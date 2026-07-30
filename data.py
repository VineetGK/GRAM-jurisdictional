"""Dataset utilities for GRAM legal LLM - loading, preprocessing, and DataLoader creation."""

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset
from datasets import load_dataset, Dataset as HFDataset

from config import config
from tokenizer import LegalBPETokenizer, get_tokenizer

# Optional HF token for gated datasets
HF_TOKEN = getattr(config, 'hf_token', None)


@dataclass
class JurisdictionBatch:
    """Batch with jurisdiction labels for GRAM routing."""
    input_ids: torch.Tensor  # (batch_size, seq_len)
    attention_mask: torch.Tensor
    labels: torch.Tensor
    jurisdiction: torch.Tensor  # (batch_size,) - 0=general, 1=US, 2=EU


class LegalTextDataset(Dataset):
    """Dataset for legal text with jurisdiction labels."""

    def __init__(
        self,
        texts: List[str],
        jurisdictions: List[str],
        tokenizer: LegalBPETokenizer,
        max_length: int = 512,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Chunk and tokenize all texts
        self.examples = []
        jur_to_id = {"general": 0, "US": 1, "EU": 2}

        for text, jur in zip(texts, jurisdictions):
            chunks = self._chunk_text(text)
            for chunk in chunks:
                if len(chunk.strip()) < 50:
                    continue
                encoded = tokenizer.encode(
                    chunk,
                    add_special_tokens=True,
                    max_length=max_length,
                    truncation=True,
                )
                if isinstance(encoded, dict):
                    input_ids = encoded["input_ids"]
                else:
                    input_ids = encoded

                if len(input_ids) > 10:
                    self.examples.append({
                        "input_ids": input_ids,
                        "jurisdiction": jur_to_id.get(jur, 0),
                    })

        print(f"Created {len(self.examples)} examples from {len(texts)} documents")

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        words = text.split()
        if len(words) <= self.chunk_size:
            return [text]

        chunks = []
        step = self.chunk_size - self.chunk_overlap
        for i in range(0, len(words), step):
            chunk = " ".join(words[i:i + self.chunk_size])
            chunks.append(chunk)
            if i + self.chunk_size >= len(words):
                break
        return chunks

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class LegalIterableDataset(IterableDataset):
    """Streaming dataset for large legal corpora."""

    def __init__(
        self,
        dataset_name: str,
        split: str,
        tokenizer: LegalBPETokenizer,
        jurisdiction: str,
        max_length: int = 512,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        num_samples: int = None,
        dataset_config: str = None,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.tokenizer = tokenizer
        self.jurisdiction = jurisdiction
        self.max_length = max_length
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.num_samples = num_samples
        self.dataset_config = dataset_config
        self.jur_to_id = {"general": 0, "US": 1, "EU": 2}

    def __iter__(self) -> Iterator[Dict]:
        ds = load_dataset(self.dataset_name, self.dataset_config, split=self.split, streaming=True, token=HF_TOKEN)
        jur_id = self.jur_to_id.get(self.jurisdiction, 0)
        count = 0

        for example in ds:
            text = example.get("text", example.get("content", ""))
            if not text or len(text.strip()) < 50:
                continue

            chunks = self._chunk_text(text)
            for chunk in chunks:
                encoded = self.tokenizer.encode(
                    chunk,
                    add_special_tokens=True,
                    max_length=self.max_length,
                    truncation=True,
                )
                if isinstance(encoded, dict):
                    input_ids = encoded["input_ids"]
                else:
                    input_ids = encoded

                if len(input_ids) > 10:
                    yield {
                        "input_ids": input_ids,
                        "jurisdiction": jur_id,
                    }
                    count += 1
                    if self.num_samples and count >= self.num_samples:
                        return

    def _chunk_text(self, text: str) -> List[str]:
        words = text.split()
        if len(words) <= self.chunk_size:
            return [text]

        chunks = []
        step = self.chunk_size - self.chunk_overlap
        for i in range(0, len(words), step):
            chunk = " ".join(words[i:i + self.chunk_size])
            chunks.append(chunk)
            if i + self.chunk_size >= len(words):
                break
        return chunks


def collate_batch(batch: List[Dict], pad_token_id: int = 0, max_length: int = 512) -> JurisdictionBatch:
    """Collate function for legal dataset batches."""
    input_ids = [torch.tensor(ex["input_ids"], dtype=torch.long) for ex in batch]
    jurisdictions = [ex["jurisdiction"] for ex in batch]

    # Pad sequences
    max_len = min(max(len(ids) for ids in input_ids), max_length)
    padded_ids = []
    attention_masks = []

    for ids in input_ids:
        if len(ids) > max_len:
            ids = ids[:max_len]
        pad_len = max_len - len(ids)
        padded_ids.append(torch.cat([ids, torch.full((pad_len,), pad_token_id, dtype=torch.long)]))
        attention_masks.append(torch.cat([torch.ones(len(ids), dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)]))

    input_ids = torch.stack(padded_ids)
    attention_mask = torch.stack(attention_masks)
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    jurisdiction = torch.tensor(jurisdictions, dtype=torch.long)

    return JurisdictionBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        jurisdiction=jurisdiction,
    )


def create_dataloaders(
    tokenizer: LegalBPETokenizer,
    batch_size: int = 16,
    num_workers: int = 4,
    us_sample_size: int = None,
    eu_sample_size: int = None,
    general_sample_size: int = None,
    max_length: int = 512,
    use_streaming: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader, DataLoader, DataLoader]:
    """
    Create train/val dataloaders for US, EU, and general data.
    Returns: (us_train, us_val, eu_train, eu_val, general_train, general_val)
    """
    us_sample = us_sample_size or config.us_sample_size
    eu_sample = eu_sample_size or config.eu_sample_size
    general_sample = general_sample_size or config.general_sample_size

    def create_loader(dataset_name, split, jur, sample_size, shuffle=True, dataset_config=None):
        if use_streaming:
            ds = LegalIterableDataset(
                dataset_name, split, tokenizer, jur,
                max_length=max_length,
                num_samples=sample_size,
                dataset_config=dataset_config,
            )
            return DataLoader(
                ds,
                batch_size=batch_size,
                num_workers=0,
                collate_fn=lambda b: collate_batch(b, tokenizer.pad_token_id, max_length),
            )
        else:
            # For non-streaming, we'd need to load into memory first
            # Using a simple approach here
            ds = load_dataset(dataset_name, dataset_config, split=split, streaming=True, token=HF_TOKEN)
            texts = []
            for i, ex in enumerate(ds):
                text = ex.get("text", ex.get("content", ""))
                if text and len(text.strip()) > 50:
                    texts.append(text)
                if len(texts) >= sample_size:
                    break

            dataset = LegalTextDataset(
                texts, [jur] * len(texts), tokenizer,
                max_length=max_length,
            )
            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                collate_fn=lambda b: collate_batch(b, tokenizer.pad_token_id, max_length),
            )

    print("Loading US Caselaw...")
    try:
        us_train = create_loader(
            "free-law/Caselaw_Access_Project", "train", "US",
            us_sample, shuffle=True
        )
        us_val = create_loader(
            "free-law/Caselaw_Access_Project", "train", "US",
            us_sample // 10, shuffle=False
        )
    except Exception as e:
        print(f"Could not load Caselaw Access Project (gated): {e}")
        print("Using dummy data for US. To use real data, authenticate with HF or use local files.")
        us_train = create_dummy_loader(tokenizer, "US", batch_size, us_sample)
        us_val = create_dummy_loader(tokenizer, "US", batch_size, us_sample // 10)

    print("Loading EU EurLex...")
    try:
        eu_train = create_loader(
            "lex_glue", "train", "EU",
            eu_sample, shuffle=True, dataset_config="eurlex"
        )
        eu_val = create_loader(
            "lex_glue", "train", "EU",
            eu_sample // 10, shuffle=False, dataset_config="eurlex"
        )
    except Exception as e:
        print(f"Could not load EUR-Lex: {e}, using dummy data")
        eu_train = create_dummy_loader(tokenizer, "EU", batch_size, eu_sample)
        eu_val = create_dummy_loader(tokenizer, "EU", batch_size, eu_sample // 10)

    print("Loading general corpus (Wikipedia)...")
    try:
        general_train = create_loader(
            "wikipedia", "train", "general",
            general_sample, shuffle=True, dataset_config="20220301.en"
        )
        general_val = create_loader(
            "wikipedia", "train", "general",
            general_sample // 10, shuffle=False, dataset_config="20220301.en"
        )
    except Exception as e:
        print(f"Could not load Wikipedia: {e}, using dummy data")
        general_train = create_dummy_loader(tokenizer, "general", batch_size, general_sample)
        general_val = create_dummy_loader(tokenizer, "general", batch_size, general_sample // 10)

    return us_train, us_val, eu_train, eu_val, general_train, general_val


def create_dummy_loader(
    tokenizer: LegalBPETokenizer,
    jurisdiction: str,
    batch_size: int,
    num_samples: int,
    max_length: int = 512,
) -> DataLoader:
    """Create dummy data loader for testing."""
    jur_to_id = {"general": 0, "US": 1, "EU": 2}
    jur_id = jur_to_id.get(jurisdiction, 0)

    texts = [
        f"This is a sample {jurisdiction} legal text for testing purposes. " * 20
        for _ in range(num_samples)
    ]

    dataset = LegalTextDataset(
        texts, [jurisdiction] * len(texts), tokenizer,
        max_length=max_length,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda b: collate_batch(b, tokenizer.pad_token_id, max_length),
    )


class MixedJurisdictionDataLoader:
    """
    DataLoader that yields batches from multiple jurisdictions for GRAM training.
    Implements alternating or mixed sampling strategy.
    """

    def __init__(
        self,
        us_loader: DataLoader,
        eu_loader: DataLoader,
        general_loader: DataLoader,
        batch_size: int,
        phase: str = "gram",  # "warmup" or "gram"
        general_ratio: float = 0.1,  # fraction of general batches
    ):
        self.us_loader = us_loader
        self.eu_loader = eu_loader
        self.general_loader = general_loader
        self.batch_size = batch_size
        self.phase = phase
        self.general_ratio = general_ratio

        self.us_iter = iter(us_loader)
        self.eu_iter = iter(eu_loader)
        self.general_iter = iter(general_loader)

    def __iter__(self):
        return self

    def __next__(self) -> JurisdictionBatch:
        if self.phase == "warmup":
            # Phase 1: mostly general data
            try:
                return next(self.general_iter)
            except StopIteration:
                self.general_iter = iter(self.general_loader)
                return next(self.general_iter)

        # Phase 2: GRAM phase - alternate US and EU, occasional general
        import random
        r = random.random()

        if r < self.general_ratio:
            try:
                return next(self.general_iter)
            except StopIteration:
                self.general_iter = iter(self.general_loader)
                return next(self.general_iter)
        elif r < 0.5 + self.general_ratio / 2:
            try:
                return next(self.us_iter)
            except StopIteration:
                self.us_iter = iter(self.us_loader)
                return next(self.us_iter)
        else:
            try:
                return next(self.eu_iter)
            except StopIteration:
                self.eu_iter = iter(self.eu_loader)
                return next(self.eu_iter)

    def set_phase(self, phase: str):
        """Switch between warmup and GRAM phase."""
        self.phase = phase


def get_dataloaders(
    tokenizer: LegalBPETokenizer,
    batch_size: int = None,
    use_streaming: bool = False,
) -> Tuple[MixedJurisdictionDataLoader, DataLoader, DataLoader]:
    """
    Get training and validation dataloaders.
    Returns: (train_loader, val_us_loader, val_eu_loader)
    """
    batch_size = batch_size or config.batch_size

    us_train, us_val, eu_train, eu_val, general_train, general_val = create_dataloaders(
        tokenizer, batch_size, use_streaming=use_streaming,
    )

    train_loader = MixedJurisdictionDataLoader(
        us_train, eu_train, general_train,
        batch_size=batch_size,
        phase="warmup",
    )

    return train_loader, us_val, eu_val


if __name__ == "__main__":
    # Test dataset loading
    print("Testing tokenizer...")
    tok = get_tokenizer()
    print(f"Tokenizer vocab size: {len(tok)}")

    print("\nTesting dummy dataloader...")
    loader = create_dummy_loader(tok, "US", 4, 100)
    batch = next(iter(loader))
    print(f"Batch input_ids shape: {batch.input_ids.shape}")
    print(f"Batch jurisdiction: {batch.jurisdiction}")
    print(f"Sample decode: {tok.decode(batch.input_ids[0].tolist())[:200]}")