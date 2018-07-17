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

    def __getitem__(self, index):
        """
        Return item at index in 0 to len(self)
        In this case a set of crops from an lf sample
        Return type is a dictionary of depth and colour arrays
        """
        idx = index // self.num_crops
        start_h, start_v = self.generate_random_crop()
        end_h = start_h + self.patch_size
        end_v = start_v + self.patch_size
        depth = torch.from_numpy(
            self.depth[idx, :, start_h:end_h, start_v:end_v, :]).float()
        colour = torch.from_numpy(
            self.colour[idx, :, start_h:end_h, start_v:end_v, :]).float()
        grid_size = self.group['colour'].attrs['shape'][1]
        sample = {'depth': depth, 'colour': colour, 'grid_size': grid_size}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        """Return the number of samples in the dataset"""
        return self.group['colour'].attrs['shape'][0] * self.num_crops

    def generate_random_crop(self):
        """Return an array of random crops indexes from given light field"""
        pixel_end = self.group['disparity'].attrs['shape'][2]
        high = pixel_end - self.patch_size

        # An array of indexes to start patch extraction from
        # Array position [0] would contain patch [0] starting location
        crop_start_indexes = np.random.random_integers(0, high, 2)

        return crop_start_indexes

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
        depth = torch.from_numpy(self.depth[index, ...]).float()
        colour = torch.from_numpy(self.colour[index, ...]).float()
        grid_size = self.group['colour'].attrs['shape'][1]
        sample = {'depth': depth, 'colour': colour, 'grid_size': grid_size}

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