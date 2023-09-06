
from typing import Any, Tuple
import pytorch_lightning as pl
import h5py
import os
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import TensorDataset, Dataset, DataLoader, random_split
from tqdm import tqdm
import scipy.constants as constants



class Load_DataModule(pl.LightningDataModule):
    def __init__(self, dataset_path: str, batch_size, frame_size: int = 1024):
        super().__init__()
        self.dataset_path = os.path.expanduser(dataset_path)
        self.batch_size = batch_size
        self.frame_size = frame_size

        self.classes = ['bpsk', 'qpsk', '8psk', 'dqpsk', 'msk', '16qam', '64qam', '256qam']

    def prepare_data(self):
        pass


    def setup(self, stage: str = None, limit_samples: int = None):
        print('Preprocessing Data...')
        with h5py.File(self.dataset_path, "r") as f:
            
            x = torch.from_numpy(f['x'][:limit_samples])
            y = torch.from_numpy(f['y'][:limit_samples]).to(torch.long)
            snr = torch.from_numpy(f['snr'][:limit_samples])

            x = x.reshape((-1, 1, self.frame_size))
            
            print("Shape of x:", x.shape)
            print("Shape of y:", y.shape)
            print("Shape of snr:", snr.shape)

        
        # Create a PyTorch Dataset
        ds_full =  TensorDataset(x, y, snr)
        

        size_of_ds_full = len(ds_full)
        print("Size of ds_full:", size_of_ds_full)


        shape_of_ds_full_first_sample = ds_full[0][0].shape  
        print("Shape of ds_full:", shape_of_ds_full_first_sample )

        #self.ds_train, self.ds_val, self.ds_test = random_split(ds_full, [0.6, 0.2, 0.2], generator = torch.Generator().manual_seed(42))
          
        self.ds_test = ds_full
        
        

      
    #def train_dataloader(self) -> DataLoader:
    #    return self._data_loader(self.ds_train, shuffle=True)

    #def val_dataloader(self) -> DataLoader:
    #    return self._data_loader(self.ds_val, shuffle=False)
    
    def test_dataloader(self) -> DataLoader:
        return self._data_loader(self.ds_test, shuffle=False)

    def _data_loader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=8,
            drop_last=False,
            pin_memory=True,
            persistent_workers=True,
            #generator=self.rng
        )

