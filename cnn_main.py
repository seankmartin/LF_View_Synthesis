import os
import argparse

import tensorflow as tf
import numpy as np
import scipy as sp
from scipy import io
from scipy import interpolate
from scipy import ndimage
import matplotlib.pyplot as plt

#Set the default base directory
home = os.path.expanduser('~')
path_args = [home, 'turing', 'overflow-storage']
base_dir = os.path.join(*path_args)

#Modifiable parameters
parser = argparse.ArgumentParser(
    description='Process modifiable parameters from command line')
parser.add_argument('--base_dir', type = str, default = base_dir,
                    help = 'base directory for the data')

#Load the data from h5 file
base_dir = args.base_dir
h5_lf_vis_file_loc = os.path.join(base_dir, 'lfvishead.h5')
mri_head_vol_file_loc = os.path.join(base_dir, 'lfvis_mri_head_vol.npy')
