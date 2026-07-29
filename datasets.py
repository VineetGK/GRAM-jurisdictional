"""
Datasets module for GRAM Legal LLM
===================================

Handles downloading, preprocessing, and DataLoader creation for:
- US law: free-law/Caselaw_Access_Project (public domain US cases)
- EU law: CEPS EurLex dataset (EU legal acts)
- General: Wikipedia (generic English)

Creates jurisdiction-labeled token chunks for GRAM training.
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Iterator, Optional, Tuple
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm

from config import config
from tokenizer import (
    load_tokenizer, 
    encode_with_jurisdiction, 
    SPECIAL_TOKENS,
    train_tokenizer,
)


@dataclass
class JurisdictionSample:
    text: str
    jurisdiction: str
    metadata: Dict = None


def clean_text(text: str) -> str:
    """Clean and normalize legal text."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\.\,\;\:\!\?\(\)\[\]\{\}\-\'\"]', '', text)
    text = text.strip()
    return text


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks of approximately chunk_size words."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if len(chunk.split()) > 50:
            chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks


def download_us_dataset(data_dir: Path, max_samples: Optional[int] = None) -> List[JurisdictionSample]:
    """Download and process US Caselaw Access Project dataset."""
    print("Loading US Caselaw dataset...")
    save_path = data_dir / "us_corpus.txt"
    
    if save_path.exists():
        print(f"Loading cached US data from {save_path}")
        with open(save_path, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
        return [JurisdictionSample(t, "US") for t in texts[:max_samples]]
    
    try:
        dataset = load_dataset(config.us_dataset_name, split="train", streaming=True)
    except Exception as e:
        print(f"Failed to load US dataset: {e}. Creating dummy data.")
        return create_dummy_us_data(max_samples)
    
    texts = []
    for i, example in enumerate(tqdm(dataset, desc="Processing US cases")):
        if max_samples and i >= max_samples:
            break
        text = example.get("text") or example.get("opinion") or str(example)
        text = clean_text(text)
        if len(text) > 100:
            texts.append(text)
    
    with open(save_path, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(t + "\n")
    
    return [JurisdictionSample(t, "US") for t in texts]


def download_eu_dataset(data_dir: Path, max_samples: Optional[int] = None) -> List[JurisdictionSample]:
    """Download and process EU EurLex dataset."""
    print("Loading EU EurLex dataset...")
    save_path = data_dir / "eu_corpus.txt"
    
    if save_path.exists():
        print(f"Loading cached EU data from {save_path}")
        with open(save_path, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
        return [JurisdictionSample(t, "EU") for t in texts[:max_samples]]
    
    try:
        dataset = load_dataset(config.eu_dataset_name, split="train", streaming=True)
    except Exception as e:
        print(f"Failed to load EU dataset: {e}. Creating dummy data.")
        return create_dummy_eu_data(max_samples)
    
    texts = []
    for i, example in enumerate(tqdm(dataset, desc="Processing EU acts")):
        if max_samples and i >= max_samples:
            break
        text = example.get("text") or example.get("content") or str(example)
        text = clean_text(text)
        if len(text) > 100:
            texts.append(text)
    
    with open(save_path, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(t + "\n")
    
    return [JurisdictionSample(t, "EU") for t in texts]


def download_general_dataset(data_dir: Path, max_samples: Optional[int] = None) -> List[JurisdictionSample]:
    """Download and process general corpus (Wikipedia)."""
    print("Loading General (Wikipedia) dataset...")
    save_path = data_dir / "general_corpus.txt"
    
    if save_path.exists():
        print(f"Loading cached general data from {save_path}")
        with open(save_path, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
        return [JurisdictionSample(t, "general") for t in texts[:max_samples]]
    
    try:
        dataset = load_dataset(
            config.general_dataset_name, 
            config.general_dataset_config, 
            split="train", 
            streaming=True
        )
    except Exception as e:
        print(f"Failed to load general dataset: {e}. Creating dummy data.")
        return create_dummy_general_data(max_samples)
    
    texts = []
    for i, example in enumerate(tqdm(dataset, desc="Processing Wikipedia")):
        if max_samples and i >= max_samples:
            break
        text = example.get("text") or str(example)
        text = clean_text(text)
        if len(text) > 100:
            texts.append(text)
    
    with open(save_path, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(t + "\n")
    
    return [JurisdictionSample(t, "general") for t in texts]


def create_dummy_us_data(max_samples: int = 1000) -> List[JurisdictionSample]:
    """Create dummy US legal data for testing."""
    templates = [
        "In the case of {party1} v. {party2}, the {court} held that {holding}.",
        "The {statute} provides that {provision}. This Court has previously held that {precedent}.",
        "Under {jurisdiction} law, the elements of {claim} are: {elements}.",
    ]
    import random
    samples = []
    for i in range(max_samples or 1000):
        template = random.choice(templates)
        text = template.format(
            party1=f"Party{random.randint(1,100)}",
            party2=f"Party{random.randint(1,100)}",
            court=random.choice(["Supreme Court", "Court of Appeals", "District Court"]),
            holding="the contract was enforceable",
            statute=random.choice(["UCC § 2-207", "42 U.S.C. § 1983", "Rule 12(b)(6)"]),
            provision="warranties cannot be disclaimed",
            precedent="implied warranties apply",
            jurisdiction=random.choice(["Federal", "California", "New York", "Delaware"]),
            claim="breach of contract",
            elements="offer, acceptance, consideration, breach, damages",
        )
        samples.append(JurisdictionSample(text, "US"))
    return samples


def create_dummy_eu_data(max_samples: int = 1000) -> List[JurisdictionSample]:
    """Create dummy EU legal data for testing."""
    templates = [
        "Pursuant to Article {article} of Regulation {regulation}, {provision}.",
        "Directive {directive} requires Member States to {requirement}.",
        "The Court of Justice held in Case C-{case} that {holding}.",
    ]
    import random
    samples = []
    for i in range(max_samples or 1000):
        template = random.choice(templates)
        text = template.format(
            article=random.randint(1, 100),
            regulation=random.choice(["2016/679 (GDPR)", "2019/1150", "1215/2012"]),
            provision="data controllers must implement appropriate safeguards",
            directive=random.choice(["2004/38/EC", "2000/31/EC", "2014/24/EU"]),
            requirement="ensure free movement of citizens",
            case=random.randint(100, 999),
            holding="the directive has direct effect",
        )
        samples.append(JurisdictionSample(text, "EU"))
    return samples


def create_dummy_general_data(max_samples: int = 1000) -> List[JurisdictionSample]:
    """Create dummy general English data for testing."""
    templates = [
        "The {topic} is a {description} that {detail}.",
        "In {year}, {event} occurred, leading to {consequence}.",
        "According to {source}, {fact}.",
    ]
    import random
    samples = []
    for i in range(max_samples or 1000):
        template = random.choice(templates)
        text = template.format(
            topic=random.choice(["contract", "tort", "property", "constitutional law", "criminal procedure"]),
            description=random.choice(["legal concept", "judicial doctrine", "statutory provision"]),
            detail="governs relationships between parties",
            year=random.randint(1900, 2024),
            event=random.choice(["a landmark decision", "legislative reform", "treaty ratification"]),
            consequence="significant legal changes",
            source=random.choice(["scholars", "the Restatement", "legal commentary"]),
            fact="the rule has evolved over time",
        )
        samples.append(JurisdictionSample(text, "general"))
    return samples


class LegalDataset(IterableDataset):
    """Iterable dataset yielding tokenized jurisdiction-labeled chunks."""
    
    def __init__(
        self,
        samples: List[JurisdictionSample],
        tokenizer,
        chunk_size: int = 512,
        max_seq_len: int = 1024,
        shuffle: bool = True,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.max_seq_len = max_seq_len
        self.shuffle = shuffle
    
    def __iter__(self) -> Iterator[Dict]:
        indices = list(range(len(self.samples)))
        if self.shuffle:
            import random
            random.shuffle(indices)
        
        for idx in indices:
            sample = self.samples[idx]
            chunks = chunk_text(sample.text, self.chunk_size)
            
            for chunk in chunks:
                encoded = encode_with_jurisdiction(
                    self.tokenizer,
                    chunk,
                    sample.jurisdiction,
                    max_length=self.max_seq_len,
                )
                yield {
                    "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
                    "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
                    "jurisdiction": sample.jurisdiction,
                    "labels": torch.tensor(encoded["input_ids"], dtype=torch.long),
                }


def collate_fn(batch: List[Dict]) -> Dict:
    """Collate variable-length sequences into padded batch."""
    max_len = max(len(item["input_ids"]) for item in batch)
    
    input_ids = []
    attention_mask = []
    labels = []
    jurisdictions = []
    
    for item in batch:
        pad_len = max_len - len(item["input_ids"])
        input_ids.append(torch.cat([item["input_ids"], torch.full((pad_len,), config.tokenizer.pad_token_id if hasattr(config, 'tokenizer') else 0)]))
        attention_mask.append(torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)]))
        labels.append(torch.cat([item["labels"], torch.full((pad_len,), -100)]))
        jurisdictions.append(item["jurisdiction"])
    
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels),
        "jurisdictions": jurisdictions,
    }


def create_dataloaders(
    tokenizer,
    batch_size: int = None,
    max_seq_len: int = None,
    chunk_size: int = None,
    num_workers: int = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """Create train/val dataloaders for US, EU, and General datasets."""
    
    batch_size = batch_size or config.batch_size
    max_seq_len = max_seq_len or config.max_seq_len
    chunk_size = chunk_size or config.chunk_size
    num_workers = num_workers or config.num_workers
    
    print("Loading datasets...")
    
    us_samples = download_us_dataset(config.data_dir, config.max_samples_us)
    eu_samples = download_eu_dataset(config.data_dir, config.max_samples_eu)
    general_samples = download_general_dataset(config.data_dir, config.max_samples_general)
    
    print(f"US samples: {len(us_samples)}")
    print(f"EU samples: {len(eu_samples)}")
    print(f"General samples: {len(general_samples)}")
    
    split_idx_us = int(len(us_samples) * 0.9)
    split_idx_eu = int(len(eu_samples) * 0.9)
    split_idx_gen = int(len(general_samples) * 0.9)
    
    us_train = LegalDataset(us_samples[:split_idx_us], tokenizer, chunk_size, max_seq_len, shuffle=True)
    us_val = LegalDataset(us_samples[split_idx_us:], tokenizer, chunk_size, max_seq_len, shuffle=False)
    eu_train = LegalDataset(eu_samples[:split_idx_eu], tokenizer, chunk_size, max_seq_len, shuffle=True)
    eu_val = LegalDataset(eu_samples[split_idx_eu:], tokenizer, chunk_size, max_seq_len, shuffle=False)
    gen_train = LegalDataset(general_samples[:split_idx_gen], tokenizer, chunk_size, max_seq_len, shuffle=True)
    gen_val = LegalDataset(general_samples[split_idx_gen:], tokenizer, chunk_size, max_seq_len, shuffle=False)
    
    def make_loader(dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=config.pin_memory,
            prefetch_factor=config.prefetch_factor if num_workers > 0 else None,
            collate_fn=collate_fn,
            drop_last=shuffle,
        )
    
    us_train_loader = make_loader(us_train, True)
    us_val_loader = make_loader(us_val, False)
    eu_train_loader = make_loader(eu_train, True)
    eu_val_loader = make_loader(eu_val, False)
    gen_train_loader = make_loader(gen_train, True)
    gen_val_loader = make_loader(gen_val, False)
    
    return (us_train_loader, us_val_loader, eu_train_loader, eu_val_loader, 
            gen_train_loader, gen_val_loader)


def create_unified_dataloader(
    tokenizer,
    batch_size: int = None,
    max_seq_len: int = None,
    chunk_size: int = None,
    num_workers: int = None,
) -> DataLoader:
    """Create a single dataloader that cycles through US, EU, General."""
    
    batch_size = batch_size or config.batch_size
    max_seq_len = max_seq_len or config.max_seq_len
    chunk_size = chunk_size or config.chunk_size
    num_workers = num_workers or config.num_workers
    
    print("Loading datasets for unified dataloader...")
    
    us_samples = download_us_dataset(config.data_dir, config.max_samples_us)
    eu_samples = download_eu_dataset(config.data_dir, config.max_samples_eu)
    general_samples = download_general_dataset(config.data_dir, config.max_samples_general)
    
    all_samples = us_samples + eu_samples + general_samples
    
    import random
    random.shuffle(all_samples)
    
    split_idx = int(len(all_samples) * 0.9)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]
    
    train_dataset = LegalDataset(train_samples, tokenizer, chunk_size, max_seq_len, shuffle=True)
    val_dataset = LegalDataset(val_samples, tokenizer, chunk_size, max_seq_len, shuffle=False)
    
    def make_loader(dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=config.pin_memory,
            prefetch_factor=config.prefetch_factor if num_workers > 0 else None,
            collate_fn=collate_fn,
            drop_last=shuffle,
        )
    
    train_loader = make_loader(train_dataset, True)
    val_loader = make_loader(val_dataset, False)
    
    return train_loader, val_loader


if __name__ == "__main__":
    tokenizer = train_tokenizer(
        data_dir=str(config.data_dir),
        vocab_size=config.tokenizer_vocab_size,
        min_frequency=config.tokenizer_min_frequency,
        output_dir=str(config.tokenizer_dir),
    )
    
    us_loader, us_val, eu_loader, eu_val, gen_loader, gen_val = create_dataloaders(tokenizer)
    
    print("\nTesting US loader:")
    for batch in us_loader:
        print(f"  input_ids: {batch['input_ids'].shape}")
        print(f"  attention_mask: {batch['attention_mask'].shape}")
        print(f"  labels: {batch['labels'].shape}")
        print(f"  jurisdictions: {batch['jurisdictions'][:3]}")
        break
    
    print("\nTesting EU loader:")
    for batch in eu_loader:
        print(f"  jurisdictions: {batch['jurisdictions'][:3]}")
        break
    
    print("\nTesting General loader:")
    for batch in gen_loader:
        print(f"  jurisdictions: {batch['jurisdictions'][:3]}")
        break