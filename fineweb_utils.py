
import os
import multiprocessing
from typing import Literal

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import polars as pl


num_cpus = multiprocessing.cpu_count()
DF_TRAIN = None  # global df to avoid loading the same parquet file multiple times; but don't load at import!
DF_VAL = None

class ParquetTokenizedDataset(Dataset):
    def __init__(
            self, 
            parquet_file: Literal["train_data.parquet", "val_data.parquet"], 
            sequence_length: int, 
            noop_token: int, 
    ):
        global DF_TRAIN, DF_VAL
        DF_TRAIN = pl.read_parquet(parquet_file) if DF_TRAIN is None and "train" in parquet_file else DF_TRAIN
        DF_VAL = pl.read_parquet(parquet_file) if DF_VAL is None and "val" in parquet_file else DF_VAL
        self.parquet_file = parquet_file
        self.sequence_length = sequence_length
        self.noop_token = noop_token

    def __len__(self):
        return len(DF_TRAIN) if "train" in self.parquet_file else len(DF_VAL)

    def __getitem__(self, idx):
        df = DF_TRAIN if "train" in self.parquet_file else DF_VAL
        tokens = torch.tensor(df['tokens'][idx], dtype=torch.long)
        
        mask = torch.ones((self.sequence_length,), dtype=torch.bool)
        if len(tokens) < self.sequence_length:
            padded = torch.empty((self.sequence_length,), dtype=torch.long).fill_(self.noop_token)
            padded[:len(tokens)] = tokens
            mask[len(tokens):] = 0
        elif len(tokens) > self.sequence_length:
            padded = tokens[:self.sequence_length]
        else:
            padded = tokens
        return padded, mask


def get_dataloader(
        batch_size: int, 
        sequence_length: int, 
        noop_token: int, 
        parquet_file: Literal["train_data.parquet", "val_data.parquet"], 
        num_workers=4,
) -> DataLoader:    
    dataset = ParquetTokenizedDataset(parquet_file, sequence_length, noop_token)
    
    return DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=("train" in parquet_file),
        num_workers=num_workers,
        pin_memory=True,
    )

def ce_loss(logits, target, mask):
    # Reshape logits and target
    logits = logits.view(-1, logits.size(-1))  # Shape: [22*1024, 50310]
    target = target.view(-1)  # Shape: [22*1024]
    mask = mask.view(-1)  # Shape: [22*1024]

    # Calculate loss
    loss = F.cross_entropy(logits, target, reduction='none')
    
    # Apply mask and calculate mean
    return loss[mask].mean()


def preprocess_and_save_dataset():
    if os.path.exists('train_data.parquet'):
        return
    df = pl.read_parquet('hf://datasets/snimu/fineweb-edu-sample-10BT-tiktokenized/gpt2/train-*.parquet')
    df.write_parquet('train_data.parquet')


def train_val_split(val_set_size: int, split_randomly: bool = False):
    # 1. Load val_set_size rows from train_data.parquet
    # 2. Overwrite val_data.parquet with them
    # 3. Remove them from train_data.parquet

    if not os.path.exists('train_data.parquet') or not os.path.exists('val_data.parquet'):
        preprocess_and_save_dataset()

    df_train = pl.read_parquet('train_data.parquet')
    df_val = pl.read_parquet('val_data.parquet')
    if split_randomly:
        df_val = df_train.sample(val_set_size)
    else:
        df_val = df_train.head(val_set_size)

    df_val.write_parquet('val_data.parquet')
    df_train = df_train.drop(df_val)
    df_train.write_parquet('train_data.parquet')
