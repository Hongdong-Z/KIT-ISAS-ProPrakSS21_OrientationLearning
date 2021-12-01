import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split
from numpy import pi
import numpy as np
import pandas as pd
import os
from PIL import Image



class OrientationsWithSymmDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, max_modes = 6):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
            on a sample.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.mean = None
        self.var = None
        self.max_modes = max_modes

    def __len__(self):
        return len(self.data_frame)
    
    def set_mean_and_std(self, mean, std):
        self.mean = mean
        self.std = std

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.data_frame.iloc[idx, 0])
        # im = skimage.io.imread(img_name)
        im = Image.open(img_name)
        if self.transform:
            im = self.transform(im)
            im = np.array(im)
        im = torch.from_numpy(im).float()
        if not self.mean == None:
            im = (im-self.mean)/self.std
        
        base_ori = map(float, self.data_frame.iloc[idx, 1].split(','))
        base_ori = torch.tensor(list(base_ori))
        # base_ori = base_ori / torch.norm(base_ori, p=2)
        n_modes = torch.tensor(self.data_frame.iloc[idx, 2])
        n_modes = int(n_modes)
        assert n_modes<=self.max_modes
        
        all_oris_padded = base_ori
        

        # all_oris_padded[0,:] = 8 * torch.ones(4)  #start tone
        # all_oris_padded[1,:] = base_ori
        # all_oris_padded.append(torch.zeros(4))  #end tone

        return [im,all_oris_padded,n_modes] 

class SimpleDataSplitDataModule(pl.LightningDataModule):
    def __init__(self, dataset, batch_size, num_workers = 0):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.fulldataset = dataset

    def setup(self, stage):
        noTrain = int(self.fulldataset.__len__()*0.75)
        noVal = int(self.fulldataset.__len__()*0.25)
        self.train_data, self.val_data = random_split(self.fulldataset, [noTrain, noVal])

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size,shuffle=True,num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size,num_workers=self.num_workers)#,shuffle=True)

    def test_dataloader(self):
        raise("Not supported")
        return DataLoader(self.test_data, batch_size=self.batch_size)