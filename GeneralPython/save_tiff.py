"""Image warping based on a disparity"""
import configparser
import os
from enum import Enum
import math
import argparse
from time import time

import h5py
from PIL import Image
import numpy as np
from skimage.transform import warp
import torch

def save_numpy_image_tiff(array, location):
    """Saves numpy image using PIL at location"""
    # Save a sample image
    os.makedirs(os.path.dirname(location), exist_ok=True)
    im = Image.fromarray(array)
    im.save(location)

def save_numpy_image(array, location):
    """Saves numpy image using PIL at location"""
    # Save a sample image
    os.makedirs(os.path.dirname(location), exist_ok=True)
    im = Image.fromarray(array.astype(np.uint8))
    im.save(location)

def main(args, config):
    hdf5_path = os.path.join(config['PATH']['output_dir'],
                             config['PATH']['hdf5_name'])
    with h5py.File(hdf5_path, mode='r', libver='latest') as hdf5_file:
        grid_size = 64
        grid_one_way = 8
        sample_index = grid_size // 2 + (grid_one_way // 2)
        depth_grp = hdf5_file['val']['disparity']

        overall_psnr_accum = (0, 0, 0)
        overall_ssim_accum = (0, 0, 0)
        for sample_num in range(args.nSamples, args.nSamples+1):
            SNUM = sample_num
            print("Working on image", SNUM)
            depth_image = np.squeeze(depth_grp['images'][SNUM, sample_index])
            print(depth_image[:,0])
            save_numpy_image_tiff(
                depth_image, 
                os.path.join(config['PATH']['output_dir'], "disparity.tiff")
            )
            save_numpy_image(
                depth_image, 
                os.path.join(config['PATH']['output_dir'], "disparity.png")
            )
            print(hdf5_file['val']['colour'].attrs.items())

if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(
        description='Process modifiable parameters from command line')
    PARSER.add_argument("--nSamples", "--n", type=int, default=1,
                        help="Number of sample images to warp")
    ARGS, UNPARSED = PARSER.parse_known_args()
    if len(UNPARSED) is not 0:
        print("Unrecognised command line argument passed")
        print(UNPARSED)
        exit(-1)

    CONFIG = configparser.ConfigParser()
    CONFIG.read(os.path.join('config', 'hdf5.ini'))
    DIRTOMAKE = os.path.join(CONFIG['PATH']['output_dir'], 'warped')
    if not os.path.exists(DIRTOMAKE):
        os.makedirs(DIRTOMAKE)
    main(ARGS, CONFIG)
