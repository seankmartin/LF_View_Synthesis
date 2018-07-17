"""Contains data transforms which can be passed to the data loader"""
import math

import numpy as np
import torch

import image_warping

#TODO can later fix to work on batches of size greater than 1
def disparity_based_rendering(disparities, views, grid_size):
    """Returns a list of warped images using the input views and disparites"""
     # Alternatively, grid_one_way - 1 can be used below
    shape = (grid_size,) + views.shape[-3:]
    warped_images = np.empty(
        shape=shape, dtype=np.uint8)
    grid_one_way = int(math.sqrt(grid_size))
    for i in range(grid_one_way):
        for j in range(grid_one_way):
            res = image_warping.fw_warp_image(
                views[grid_size // 2 + (grid_one_way // 2), ...],
                disparities[grid_size // 2 + (grid_one_way // 2), ...],
                np.asarray([grid_one_way // 2, grid_one_way // 2]),
                np.asarray([i, j])
            )
            np.insert(warped_images, i * grid_one_way + j, res, axis=0)
    return warped_images

def transform_to_warped(sample):
    """
    Input a dictionary of depth images and reference views,
    Output a dictionary of inputs -warped and targets - reference
    """

    disparity = sample['depth']
    targets = sample['colour']
    grid_size = sample['grid_size']
    warped_images = disparity_based_rendering(
        disparity.numpy(), targets.numpy(), grid_size)
    inputs = torch.from_numpy(warped_images).float()
    return {'inputs': inputs, 'targets': targets}