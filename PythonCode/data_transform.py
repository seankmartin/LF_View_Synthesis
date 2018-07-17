"""Contains data transforms which can be passed to the data loader"""
import math

import numpy as np
import torch

import image_warping

def disparity_based_rendering(disparities, views, grid_size, dtype=np.float32):
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
                dtype=dtype
            )
            np.insert(warped_images, i * grid_one_way + j, res, axis=0)
    return warped_images

def transform_to_warped(sample):
    """
    Input a dictionary of depth images and reference views,
    Output a dictionary of inputs -warped and targets - reference
    """
    sample = normalise_lf(sample)
    disparity = sample['depth']
    targets = sample['colour']
    grid_size = sample['grid_size']
    warped_images = disparity_based_rendering(
        disparity.numpy(), targets.numpy(), grid_size)
    inputs = torch.from_numpy(warped_images).float()
    return {'inputs': inputs, 'targets': targets}

def normalise_lf(sample):
    """Coverts an lf in the range 0 to maximum into -1 1"""
    maximum = 255.0
    lf = sample['colour']
    ((lf.div_(maximum)).mul_(2.0)).add_(-1.0)
    return sample