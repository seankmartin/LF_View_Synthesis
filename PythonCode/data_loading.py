import random

import numpy as np
import torch
import torch.utils.data as data

import data_transform
from torch.utils.data import DataLoader

class TrainFromHdf5(data.Dataset):
    """
    Creates a training set from a hdf5 file
    """
    def __init__(self, hdf_file, patch_size, num_crops, transform=None):
        """
        Keyword arguments:
        hdf_file -- the location containing the hdf5 file
        patch_size -- the size of the patches to extract for training
        num_crops -- the number of patches to extract for training
        transform -- an optional transform to apply to the data
        """
        super()
        self.group = hdf_file['train']
        self.depth = self.group['disparity']['images']
        self.colour = self.group['colour']['images']
        self.transform = transform
        self.patch_size = patch_size
        self.num_crops = num_crops
        random.seed()

    def __getitem__(self, index):
        """
        Return item at index in 0 to len(self)
        In this case a set of crops from an lf sample
        Return type is a dictionary of depth and colour arrays
        """
        idx = index // self.num_crops
        depth = self.depth[idx]
        colour = self.colour[idx]
        grid_size = self.group['colour'].attrs['shape'][1]
        sample = {'depth': depth, 'colour': colour, 'grid_size': grid_size}
        
        sample = data_transform.get_random_crop(sample, self.patch_size)

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        """Return the number of samples in the dataset"""
        return self.group['colour'].attrs['shape'][0] * self.num_crops

class ValFromHdf5(data.Dataset):
    """
    Creates a validation set from a hdf5 file
    """
    def __init__(self, hdf_file, transform=None):
        """
        Keyword arguments:
        hdf_file -- the location containing the hdf5 file
        transform -- an optional transform to apply to the data
        """
        super()
        self.group = hdf_file['val']
        self.depth = self.group['disparity']['images']
        self.colour = self.group['colour']['images']
        self.transform = transform

    def __getitem__(self, index):
        """
        Return item at index in 0 to len(self)
        In this case a set of crops from an lf sample
        Return type is a dictionary of depth and colour arrays
        """
        depth = torch.tensor(self.depth[index, ...], dtype=torch.float32)
        colour = torch.tensor(self.colour[index, ...], dtype=torch.float32)
        grid_size = self.group['colour'].attrs['shape'][1]
        sample = {'depth': depth, 'colour': colour, 'grid_size': grid_size}
        
        # Running out of GPU memory on validation
        sample = data_transform.upper_left_patch(sample)

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        """Return the number of samples in the dataset"""
        return self.group['colour'].attrs['shape'][0]

def create_dataloaders(hdf_file, args, config):
    """Creates a train and val dataloader from a h5file and a config file"""
    print("Loading dataset")
    train_set = TrainFromHdf5(
        hdf_file=hdf_file,
        patch_size=int(config['NETWORK']['patch_size']),
        num_crops=int(config['NETWORK']['num_crops']),
        transform=data_transform.transform_to_warped)
    val_set = ValFromHdf5(
        hdf_file=hdf_file,
        transform=data_transform.transform_to_warped)

    batch_size = {'train': int(config['NETWORK']['batch_size']), 'val': 1}
    data_loaders = {}
    for name, dset in (('train', train_set), ('val', val_set)):
        data_loaders[name] = DataLoader(
            dataset=dset, num_workers=args.threads,
            batch_size=batch_size[name],
            shuffle=True)

    return data_loaders