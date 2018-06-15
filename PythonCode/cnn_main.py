import os
import argparse
import configparser

import h5py
import numpy as np

def load_data(data_path):
    """Returns an array of images from the hdf5 file in data_path"""
    h5f = h5py.File(data_path, 'r')
    all_images = h5f['images']
    return all_images

def main(args, config):
    if args.base_dir == "":
        #Set the default base directory
        home = os.path.expanduser('~')
        path_args = [home, 'turing', 'overflow-storage']
        base_dir = os.path.join(*path_args)

    #Load the data
    h5_lf_vis_file_loc = os.path.join(base_dir, 'lfvishead.h5')
    images = load_data(h5_lf_vis_file_loc)

    #Perform data augmentation

    #Load the appropriate model with gpu support

    #Perform training and testing [evaluation could go in seperate script]
        #Store logs and checkpoints of models

if __name__ == '__main__':
    #Command line modifiable parameters
    parser = argparse.ArgumentParser(
        description='Process modifiable parameters from command line')
    parser.add_argument('--base_dir', type = str, default = "",
                        help = 'base directory for the data')
    args, unparsed = parser.parse_known_args()

    #Config file modifiable parameters
    config = configparser.ConfigParser()
    config.read('main_config.ini')
    
    main(args, config)
