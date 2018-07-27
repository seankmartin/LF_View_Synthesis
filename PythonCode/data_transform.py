"""Contains data transforms which can be passed to the data loader"""
import math
import random

import numpy as np
import torch

import image_warping

def disparity_based_rendering(
        disparities, views, grid_size,
        dtype=np.float32, blank=-1.0):
    """Returns a list of warped images using the input views and disparites"""
     # Alternatively, grid_one_way - 1 can be used below
    shape = (grid_size,) + views.shape[-3:]
    warped_images = np.empty(
        shape=shape, dtype=dtype)
    grid_one_way = int(math.sqrt(grid_size))
    sample_index = grid_size // 2 + (grid_one_way // 2)
    for i in range(grid_one_way):
        for j in range(grid_one_way):
            res = image_warping.fw_warp_image(
                ref_view=views[sample_index, ...],
                disparity_map=disparities[sample_index, ...],
                ref_pos=np.asarray([grid_one_way // 2, grid_one_way // 2]),
                novel_pos=np.asarray([i, j]),
                dtype=dtype,
                blank=blank
            )
            warped_images[i * grid_one_way + j] = res
    return warped_images

def transform_to_warped(sample):
    """
    Input a dictionary of depth images and reference views,
    Output a dictionary of inputs -warped and targets - reference
    """
    normalise_sample(sample)
    disparity = sample['depth']
    targets = sample['colour']
    grid_size = sample['grid_size']
    warped_images = disparity_based_rendering(
        disparity.numpy(), targets.numpy(), grid_size,
        dtype=np.float32, blank=-1.0)

    inputs = torch.from_numpy(warped_images)
    return {'inputs': inputs, 'targets': targets}

def normalise_sample(sample):
    """Coverts an lf in the range 0 to maximum into -1 1"""
    maximum = 255.0
    lf = sample['colour']
    ((lf.div_(maximum)).mul_(2.0)).add_(-1.0)
    return sample

def upper_left_patch(sample):
    width = sample['colour'].shape[2]
    sample['colour'] = sample['colour'][:, 0:width//2, 0:width//2, :]
    sample['depth'] = sample['depth'][:, 0:width//2, 0:width//2, :]
    return sample

def get_random_crop(sample, patch_size):
    pixel_end = sample['colour'].shape[1]
    high = pixel_end - 1 - patch_size
    start_h = random.randint(0, high)
    start_v = random.randint(0, high)
    end_h = start_h + patch_size
    end_v = start_v + patch_size
    sample['depth'] = sample['depth'][:, start_h:end_h, start_v:end_v, :]
    sample['colour'] = sample['colour'][:, start_h:end_h, start_v:end_v, :]
    return sample

def denormalise_lf(lf):
    """Coverts an lf in the range 0 to maximum into -1 1"""
    maximum = 255.0
    lf.add_(1.0).div_(2.0).mul_(maximum)
    return lf