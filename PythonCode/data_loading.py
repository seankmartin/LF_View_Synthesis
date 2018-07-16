import numpy.random.uniform as uniform_rand
import torch
import torch.utils.data as data

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
        self.depth = self.group['depth']
        self.colour = self.group['colour']
        self.transform = transform
        self.patch_size = patch_size
        self.num_crops = num_crops

    def __getitem__(self, index):
        """
        Return item at index in 0 to len(self)
        In this case a set of crops from an lf sample
        Return type is a dictionary of depth and colour arrays
        """
        crop_indexes = self.generate_random_crop()
        sample = {'depth': [], 'colour': []}
        for start_h, start_v in crop_indexes:
            end_h = start_h + self.patch_size
            end_v = start_v + self.patch_size
            depth = torch.from_numpy(
                self.depth[index, :, start_h:end_h, start_v:end_v, :]).float()
            colour = torch.from_numpy(
                self.colour[index, :, start_h:end_h, start_v:end_v, :]).float()
            sample['depth'].append(depth)
            sample['colour'].append(colour)

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        """Return the number of samples in the dataset"""
        return self.depth.attrs['shape'][0]

    def generate_random_crop(self):
        """Return an array of random crops indexes from given light field"""
        pixel_end = self.depth.attrs['shape'][2]
        high = pixel_end - self.patch_size

        # An array of indexes to start patch extraction from
        # Array position [0] would contain patch [0] starting location
        crop_start_indexes = uniform_rand(0, high, (self.num_crops, 2))

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
        self.depth = self.group['depth']
        self.colour = self.group['colour']
        self.transform = transform

    def __getitem__(self, index):
        """
        Return item at index in 0 to len(self)
        In this case a set of crops from an lf sample
        Return type is a dictionary of depth and colour arrays
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
