import os
import h5py
import lightning as pl
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split, ConcatDataset
import glob

class HDF5Dataset(Dataset):
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        
        with h5py.File(self.file_path, 'r') as hf:
            if len(hf["tensor"].shape) == 4:
               self.length = len(hf['tensor'])*2  # assuming 'data' is your dataset key
            else:
                self.length = len(hf["tensor"])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as hf:
            # Assuming data and grid are stored in arrays within the HDF5 file
            print(hf["tensor"].shape)
            if len(hf["tensor"].shape) == 4:
                data = np.array(hf['tensor'][idx//5000][idx%5000])
            else:
                data = np.array(hf['tensor'][idx])
            grid = np.array(hf['x-coordinate'])  # Adjust keys as necessary
        # You may need to reshape or process data and grid here if necessary
        return data, grid

class SimpleDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, train_val_split=0.8):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_val_split = train_val_split
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None

    def prepare_data(self):
        # Check data availability, download, or preprocess if needed
        pass

    def setup(self, stage=None):
        # Load dataset
        # Assuming there's one HDF5 file for simplicity; adapt if there are multiple
        data_files = glob.glob(os.path.join(self.data_dir, "*"))
        all_dataset = []
        for hdf5_file in data_files:
            all_dataset.append(HDF5Dataset(hdf5_file))
        # hdf5_file = os.path.join(self.data_dir, 'your_data_file.h5')
        self.dataset = ConcatDataset(all_dataset)

        # Split dataset
        train_size = int(len(self.dataset) * self.train_val_split)
        val_size = len(self.dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(self.dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

if __name__ == "__main__":

    data_dir = "/mnt/f/data/PDEbench/1D/Advection/Train"

    dm = SimpleDataModule(data_dir)

    dm.setup()

    train_dl = iter(dm.train_dataloader())

    sample = next(train_dl)

    # print(sample[0].shape)
    # print(sample[1].shape)





