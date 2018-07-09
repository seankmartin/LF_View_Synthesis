import h5py

import torch
import torch.utils.data as data


class DatasetFromHdf5(data.Dataset):
    def __init__(self, file_path, transform = None):
        """
        Keyword arguments:
        file_path -- the location containing the hdf5 file
        transform -- an optional transform to apply to the data
        """
        super()
        #Need to close the file on destruction
        self.hf = h5py.File(file_path, mode = 'r', libver = 'latest')
        self.depth = self.hf['depth']
        self.colour = self.hf['colour']
        self.transform = transform

    def __getitem__(self, index):
        """
        Return item at index in 0 to len(self)
        In this case an LF sample
        Return type is a dictionary of depth and colour
        """
        depth = torch.from_numpy(self.depth[index, ...]).float()
        colour = torch.from_numpy(self.colour[index, ...]).float()
        sample = {'depth': depth, 'colour': colour}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        """Return the number of samples in the dataset"""
        return self.depth.attrs['shape'][0]

    def close_h5(self):
        self.h5.close()
