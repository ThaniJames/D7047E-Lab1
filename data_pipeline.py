# Shared Data Pipeline
# Loads Amazon Polarity (~1GB) from Hugging Face, splits, and prepares for models.

import re
import gc
import random
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

from config import (
    SEED, SPLIT_RATIOS,
    HF_DATASET_NAME, MAX_LENGTH_TRANSFORMER,
)


# Utilities
def set_seed(seed=SEED):
    """Set seed for reproducibility."""
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clean_text(text):
    """Light text cleaning for transformers."""
    if not text:
        return ""
    text = str(text)
    text = re.sub(r"<[^>]+>", " ", text)        # remove HTML tags
    text = re.sub(r"http\S+|www\S+", " ", text)  # remove URLs
    text = re.sub(r"\S+@\S+", " ", text)         # remove emails
    text = re.sub(r"\s+", " ", text)             # normalize whitespace
    return text.strip()



# Layer 1: Load datasets
def load_given_dataset(filepath):
    """Load the given 1K or 25K dataset (tab-separated .txt)."""
    import pandas as pd
    df = pd.read_csv(filepath, delimiter="\t", header=None,
                     names=["Sentence", "Class"])
    texts = [clean_text(str(t)) for t in df["Sentence"].tolist()]
    labels = df["Class"].astype(int).tolist()
    return texts, labels


def load_amazon_polarity(subset_size=None, seed=SEED):
    """Load Amazon Polarity dataset from Hugging Face"""
   

    ds = load_dataset(HF_DATASET_NAME, split="train")

    if subset_size is not None:
        ds = ds.shuffle(seed=seed).select(range(min(subset_size, len(ds))))

    # Labels are already 0=negative, 1=positive
    texts = [clean_text(row["content"]) for row in ds]
    labels = [row["label"] for row in ds]

    del ds
    gc.collect()

    return texts, labels


# Layer 1: Split
def split_data(texts, labels, split_ratios=SPLIT_RATIOS, seed=SEED):
    """Deterministic stratified split into train/val/test."""
    train_ratio, val_ratio, test_ratio = split_ratios

    # Step 1: split off test set
    temp_texts, test_texts, temp_labels, test_labels = train_test_split(
        texts, labels,
        test_size=test_ratio,
        random_state=seed,
        stratify=labels,
    )

    # Step 2: split remainder into train and val
    val_fraction = val_ratio / (train_ratio + val_ratio)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        temp_texts, temp_labels,
        test_size=val_fraction,
        random_state=seed,
        stratify=temp_labels,
    )

    return {
        "train": {"texts": train_texts, "labels": train_labels},
        "val":   {"texts": val_texts,   "labels": val_labels},
        "test":  {"texts": test_texts,  "labels": test_labels},
    }


def load_and_split(subset_size=None, filepath=None, seed=SEED, split_ratios=SPLIT_RATIOS):
    """Main entry point: load dataset and split it"""
    if filepath is not None:
        texts, labels = load_given_dataset(filepath)
    else:
        texts, labels = load_amazon_polarity(subset_size=subset_size, seed=seed)
    return split_data(texts, labels, split_ratios=split_ratios, seed=seed)



# Layer 2: Transformer preparation
def prepare_transformer_dataset(texts, labels, tokenizer_name,
                                max_length=MAX_LENGTH_TRANSFORMER):
    """Tokenize texts for a Hugging Face transformer model"""

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    ds = Dataset.from_dict({"text": texts, "labels": labels})

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        )

    ds = ds.map(tokenize_fn, batched=True, batch_size=1000)
    ds = ds.remove_columns(["text"])
    ds.set_format("torch")

    return ds
