import torch
from torch.utils.data import IterableDataset, DataLoader
import random
import numpy as np


class TLMDataset(IterableDataset):
    def __init__(self, cfg, dataset_path):
        self.dataset_path = dataset_path
        data = np.memmap(self.dataset_path, dtype=np.uint16, mode="r")
        self.chunk_size = cfg.model.chunk_size
        assert self.chunk_size % 2 == 0 # Odd chunk sizes are wierd for debug mode
        self.num_chunks = len(data) // self.chunk_size
        self.debug = cfg.train.debug

        def __iter__(self):
            data = np.memmap(self.dataset_path, dtype=np.uint16, mode="r")
            indices = np.arange(self.num_chunks)
            np.random.shuffle(indices)
            for i in indices:
                chunk = torch.from_numpy(data[i * self.chunk_size:(i + 1) * self.chunk_size].astype(np.int64))
                if self.debug:
                    inp = chunk
                    half_chunk = chunk[:self.chunk_size // 2]
                    targ = torch.cat((half_chunk, half_chunk))
                else:
                    inp = chunk[:-1]
                    targ = chunk[1:]
                yield [inp, targ]


def create_infinite_dataloader(dataloader):
    epoch = 0
    while True:
        for batch in dataloader:
            yield batch.append(epoch)
        epoch += 1


def create_dataloaders(cfg):
    train_dataset = TLMDataset(cfg, cfg.tokenization.tokenized_train)
    valid_dataset = TLMDataset(cfg, cfg.tokenization.tokenized_valid)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.train.batch_size, num_workers=0)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=cfg.train.batch_size, num_workers=0)
    train_dataloader = create_infinite_dataloader(train_dataloader)
    valid_dataloader = create_infinite_dataloader(valid_dataloader)

    return train_dataloader, valid_dataloader