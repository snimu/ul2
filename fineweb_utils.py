
import os
import multiprocessing
from typing import Literal

import torch
from torch.utils.data import Dataset, DataLoader
import polars as pl


num_cpus = multiprocessing.cpu_count()
DF = None  # global df to avoid loading the same parquet file multiple times; but don't load at import!


class ParquetTokenizedDataset(Dataset):
    def __init__(
            self, 
            parquet_file: Literal["train_data.parquet", "val_data.parquet"], 
            sequence_length: int, 
            noop_token: int, 
            device='cuda',
    ):
        global DF
        DF = pl.read_parquet(parquet_file) if DF is None else DF
        self.sequence_length = sequence_length
        self.noop_token = noop_token
        self.device = device

    def __len__(self):
        return len(DF)

    def __getitem__(self, idx):
        tokens = torch.tensor(DF['tokens'][idx], dtype=torch.long, device=self.device)
        
        mask = torch.ones((self.sequence_length,), dtype=torch.bool, device=self.device)
        if len(tokens) < self.sequence_length:
            padded = torch.empty((self.sequence_length,), dtype=torch.long, device=self.device).fill_(self.noop_token)
            padded[:len(tokens)] = tokens
            mask[:len(tokens)] = 0
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
        device='cuda',
) -> DataLoader:    
    dataset = ParquetTokenizedDataset(parquet_file, sequence_length, noop_token, device)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=("train" in parquet_file),
        num_workers=num_workers,
        pin_memory=True,
    )

def ce_loss(logits, target, mask):
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target.flatten(0, 1), reduction='none')
    return loss[mask].mean()


def preprocess_and_save_dataset():
    if os.path.exists('train_data.parquet') and os.path.exists('val_data.parquet'):
        return
    splits = {'val': 'gpt2/val-00000-of-00001.parquet', 'train': 'gpt2/train-*.parquet'}
    df = pl.read_parquet('hf://datasets/snimu/fineweb-edu-sample-10BT-tiktokenized/' + splits['train'])
    df.write_parquet('train_data.parquet')
    del df
    df = pl.read_parquet('hf://datasets/snimu/fineweb-edu-sample-10BT-tiktokenized/' + splits['val'])
    df.write_parquet('val_data.parquet')
