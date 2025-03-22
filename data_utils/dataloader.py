from typing import Iterator, List, Tuple, Any

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset
from omegaconf import DictConfig


class TLMDataset(IterableDataset):
    """Dataset class for loading and processing tokenized language model data.
    
    This class implements an IterableDataset that loads data from a memory-mapped file
    and yields chunks of tokens for training or validation.
    """
    
    def __init__(self, cfg: DictConfig, dataset_path: str):
        """Initialize the dataset.
        
        Args:
            cfg: Configuration object containing model and training parameters
            dataset_path: Path to the memory-mapped tokenized dataset
        """
        self.dataset_path = dataset_path
        data = np.memmap(self.dataset_path, dtype=np.uint16, mode="r")
        self.chunk_size = cfg.model.chunk_size
        if self.chunk_size % 2 != 0:
            raise ValueError("Chunk size must be even for proper debug mode operation")
        self.num_chunks = len(data) // self.chunk_size
        self.debug = cfg.train.debug

    def __iter__(self) -> Iterator[List[torch.Tensor]]:
        """Iterate over the dataset, yielding chunks of input and target tokens.
        
        Yields:
            List containing input and target tensors for each chunk
        """
        data = np.memmap(self.dataset_path, dtype=np.uint16, mode="r")
        indices = np.arange(self.num_chunks)
        np.random.shuffle(indices)
        
        for i in indices:
            chunk = torch.from_numpy(
                data[i * self.chunk_size:(i + 1) * self.chunk_size].astype(np.int64)
            )
            
            if self.debug:
                # In debug mode, duplicate half the chunk for target. This is meant to be a implementation sanity check
                inp = chunk
                half_chunk = chunk[:self.chunk_size // 2]
                targ = torch.cat((half_chunk, half_chunk))
            else:
                # Normal mode: target is shifted by one position
                inp = chunk[:-1]
                targ = chunk[1:]
                
            yield [inp, targ]


def create_infinite_dataloader(dataloader: DataLoader) -> Iterator[List[Any]]:
    """Create an infinite iterator over a dataloader that includes epoch information.
    
    Args:
        dataloader: PyTorch DataLoader to iterate over
        
    Yields:
        List containing batch data and current epoch number
    """
    epoch = 0
    while True:
        for batch in dataloader:
            yield batch + [epoch]
        epoch += 1


def create_dataloaders(cfg: DictConfig) -> Tuple[Iterator[List[Any]], Iterator[List[Any]]]:
    """Create training and validation dataloaders.
    
    Args:
        cfg: Configuration object containing dataset and training parameters
        
    Returns:
        Tuple of (train_dataloader, valid_dataloader) as infinite iterators
    """
    train_dataset = TLMDataset(cfg, cfg.tokenization.tokenized_train)
    valid_dataset = TLMDataset(cfg, cfg.tokenization.tokenized_valid)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        num_workers=0
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=cfg.train.batch_size,
        num_workers=0
    )
    
    return (
        create_infinite_dataloader(train_dataloader),
        create_infinite_dataloader(valid_dataloader)
    )
