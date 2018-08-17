"""Contains data transforms which can be passed to the data loader"""
import math
import random

import numpy as np
import torch
import matplotlib.cm as cm

import image_warping

def disparity_based_rendering(disparities, views, grid_size, sample_index):
    """Returns a list of warped images using the input views and disparites"""
     # Alternatively, grid_one_way - 1 can be used below
    shape = (grid_size,) + views.shape[-3:]
    return image_warping.depth_rendering(
        ref_view=views[sample_index],
        disparity_map=disparities[sample_index],
        lf_size=shape)

def transform_to_warped(sample):
    """
    Input a dictionary of depth images and reference views,
    Output a dictionary of inputs -warped and targets - reference
    """
    normalise_sample(sample)
    disparity = sample['depth']
    targets = sample['colour']
    grid_size = sample['grid_size']
    grid_one_way = int(math.sqrt(grid_size))
    sample_index = grid_size // 2 + (grid_one_way // 2)
    shape = targets.shape
    warped_images = disparity_based_rendering(
        disparity.numpy(), targets.numpy(), grid_size, sample_index)
    inputs = create_remap(warped_images, dtype=torch.float32)
    targets = create_remap(targets, dtype=torch.float32)
    return {'inputs': inputs, 'targets': targets, 'shape': shape}

def torch_stack(input_t, channels=64):
    #This has shape im_size, im_size, num_colours * num_images
    out = torch.squeeze(
        torch.cat(torch.chunk(input_t, channels, dim=0), dim=-1))

    #This has shape num_colour * num_images, im_size, im_size
    out = out.transpose_(0, 2).transpose_(1, 2)
    return out

def torch_unstack(input_t, channels=64):
    #This has shape num_images, num_channels, im_size, im_size
    input_back = torch.stack(torch.chunk(input_t, 64, dim=0))

    #This has shape num_images, im_size, im_size, num_channels
    input_back = input_back.transpose_(1, 3).transpose_(1, 2)

    return input_back

def stack(sample, channels=64):
    sample['inputs'] = torch_stack(sample['inputs'], channels)
    sample['targets'] = torch_stack(sample['targets'], channels)
    return sample

def normalise_sample(sample):
    """Coverts an lf in the range 0 to maximum into 0 1"""
    maximum = 255.0
    lf = sample['colour']
    lf.div_(maximum)
    return sample

def upper_left_patch(sample):
    width = sample['colour'].shape[2]
    sample['colour'] = sample['colour'][:, 0:width//2, 0:width//2]
    sample['depth'] = sample['depth'][:, 0:width//2, 0:width//2]
    return sample

def get_random_crop(sample, patch_size):
    pixel_end = sample['colour'].shape[1]
    high = pixel_end - 1 - patch_size
    start_h = random.randint(0, high)
    start_v = random.randint(0, high)
    end_h = start_h + patch_size
    end_v = start_v + patch_size
    sample['depth'] = sample['depth'][:, start_h:end_h, start_v:end_v]
    sample['colour'] = sample['colour'][:, start_h:end_h, start_v:end_v]
    return sample

def random_gamma(sample):
    maximum = 255
    gamma = random.uniform(0.4, 1.0)
    sample['colour'] = torch.pow(
        sample['colour'].div_(maximum), gamma).mul_(maximum)
    return sample

def denormalise_lf(lf):
    """Coverts an lf in the range 0 1 to 0 to maximum"""
    maximum = 255.0
    lf.mul_(maximum)
    return lf

def disparity_to_rgb(disparity_map):
    """Converts a disparity map into the range 0 1"""
    depth = disparity_map
    min = float(depth.min())
    max = float(depth.max())
    depth.add_(-min).div_(max - min + 1e-5)
    scale = cm.ScalarMappable(None, cmap="viridis")
    coloured = scale.to_rgba(depth, norm=False)
    return torch.tensor(coloured[:, :, :3], dtype=torch.float32) 

def create_remap(in_tensor, dtype=torch.uint8):
    """
    Remaps an input tensor of shape B, W, H, C into
    W * sqrt(B), H * sqrt(B), C
    Where each grid of size sqrt(B) * sqrt(B) in the remap
    contains pixel information from the 64 input images
    """
    num, im_width, im_height, channels = in_tensor.shape
    one_way = int(math.floor(math.sqrt(num)))
    out_tensor = torch.zeros(
        size=(im_width * one_way, im_height * one_way, channels), 
        dtype=dtype)
    for i in range(num):
        out_tensor[i//one_way::one_way, i%one_way::one_way] = in_tensor[i]
    return out_tensor.transpose_(0, 2).transpose_(1, 2)
        
def undo_remap(in_tensor, desired_shape, dtype=torch.uint8):
    """
    Remaps an input tensor to undo create_remap
    """
    num, im_width, im_height, channels = desired_shape
    one_way = int(math.floor(math.sqrt(num)))
    in_tensor.transpose_(0, 2).transpose_(1, 2)
    out_tensor = torch.zeros(size=desired_shape, dtype=dtype)
    for i in range(num):
        out_tensor[i] = in_tensor[i//one_way::one_way, i%one_way::one_way]
    return out_tensor