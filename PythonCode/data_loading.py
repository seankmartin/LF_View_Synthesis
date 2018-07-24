import random
import os

import h5py
import torch
import torch.utils.data as data

import data_transform
from torch.utils.data import DataLoader

class TrainFromHdf5(data.Dataset):
    """
    Creates a training set from a hdf5 file
    """
    def __init__(self, file_path, patch_size, num_crops, transform=None):
        """
        Keyword arguments:
        hdf_file -- the location containing the hdf5 file
        patch_size -- the size of the patches to extract for training
        num_crops -- the number of patches to extract for training
        transform -- an optional transform to apply to the data
        """
        super()
        self.file_path = file_path
        with h5py.File(
                file_path, mode='r', libver='latest', swmr=True) as h5_file:
            self.num_samples = h5_file['train/colour'].attrs['shape'][0]
            self.grid_size = h5_file['train/colour'].attrs['shape'][1]
        self.depth = '/train/disparity/images'
        self.colour = '/train/colour/images'
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
        with h5py.File(
                self.file_path, mode='r',
                libver='latest', swmr=True) as h5_file:
            idx = index // self.num_crops
            depth = torch.tensor(
                h5_file[self.depth][idx], dtype=torch.float32)
            colour = torch.tensor(
                h5_file[self.colour][idx], dtype=torch.float32)
            grid_size = self.grid_size
            sample = {'depth': depth, 'colour': colour, 'grid_size': grid_size}

            sample = data_transform.get_random_crop(sample, self.patch_size)

            if self.transform:
                sample = self.transform(sample)

            return sample

    def __len__(self):
        """Return the number of samples in the dataset"""
        return self.num_samples * self.num_crops

class ValFromHdf5(data.Dataset):
    """
    Creates a validation set from a hdf5 file
    """
    def __init__(self, file_path, transform=None):
        """
        Keyword arguments:
        hdf_file -- the location containing the hdf5 file
        transform -- an optional transform to apply to the data
        """
        super()
        self.file_path = file_path
        with h5py.File(
                file_path, mode='r', libver='latest', swmr=True) as h5_file:
            self.num_samples = h5_file['val/colour'].attrs['shape'][0]
            self.grid_size = h5_file['val/colour'].attrs['shape'][1]
        self.depth = '/val/disparity/images'
        self.colour = '/val/colour/images'
        self.transform = transform

    def __getitem__(self, index):
        """
        Return item at index in 0 to len(self)
        In this case a set of crops from an lf sample
        Return type is a dictionary of depth and colour arrays
        """
        with h5py.File(
                self.file_path, mode='r',
                libver='latest', swmr=True) as h5_file:
            depth = torch.tensor(
                h5_file[self.depth][index], dtype=torch.float32)
            colour = torch.tensor(
                h5_file[self.colour][index], dtype=torch.float32)
            grid_size = self.grid_size
            sample = {'depth': depth, 'colour': colour, 'grid_size': grid_size}

        # Running out of GPU memory on validation
        sample = data_transform.upper_left_patch(sample)

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        """Return the number of samples in the dataset"""
        return self.num_samples

def create_dataloaders(args, config):
    """Creates a train and val dataloader from a h5file and a config file"""
    print("Loading dataset")
    file_path = os.path.join(config['PATH']['hdf5_dir'],
                             config['PATH']['hdf5_name'])
    train_set = TrainFromHdf5(
        file_path=file_path,
        patch_size=int(config['NETWORK']['patch_size']),
        num_crops=int(config['NETWORK']['num_crops']),
        transform=data_transform.transform_to_warped)
    val_set = ValFromHdf5(
        file_path=file_path,
        transform=data_transform.transform_to_warped)

    batch_size = {'train': int(config['NETWORK']['batch_size']), 'val': 1}
    data_loaders = {}
    threads = int(config['NETWORK']['num_workers'])
    for name, dset in (('train', train_set), ('val', val_set)):
        data_loaders[name] = DataLoader(
            dataset=dset, num_workers=threads,
            batch_size=batch_size[name],
            shuffle=True)

    return data_loaders
