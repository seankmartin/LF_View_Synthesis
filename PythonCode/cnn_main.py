#Many things based on https://github.com/twtygqyy/pytorch-vdsr
import os
import argparse
import configparser
import warnings

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data_loading import DatasetFromHdf5
import helpers

def main(args, config):
    #Preliminary setup
    cuda = config['Network']['cuda'] == True
    if cuda:
        print("=> use gpu id: '{}'".format(opt.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
        if not torch.cuda.is_available():
                raise Exception("No GPU found or Wrong gpu id")
        print("cudnn version is", torch.backends.cudnn.version())

    #Attempts to otimise - see
    #https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do
    torch.backends.cudnn.benchmark = True

    #Load the data
    print("Loading dataset")
    #There should be a train and val here - this can be done with the h5
    h5_lf_vis_file_loc = os.path.join(config['PATH']['hdf5_dir'],
                                      config['PATH']['hdf5_name'])
    dataset = DatasetFromHdf5(h5_lf_vis_file_loc)
    data_loader = DataLoader(dataset = dataset, num_workers = args.threads,
                             batch_size = int(config['NETWORK']['batch_size']),
                             shuffle = True)
    #Load the appropriate model with gpu support
    print("Building model")

    #Perform training and testing [evaluation could go in seperate script]
        #Store logs and checkpoints of models

if __name__ == '__main__':
    #Command line modifiable parameters
    #See https://github.com/twtygqyy/pytorch-vdsr/blob/master/main_vdsr.py
    #For the source of some of these arguments
    parser = argparse.ArgumentParser(
        description='Process modifiable parameters from command line')
    parser.add_argument('--base_dir', type = str, default = "",
                        help = 'base directory for the data')
    parser.add_argument("--nEpochs", type=int, default=50,
                        help="Number of epochs to train for")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="Learning Rate. Default=0.1")
    parser.add_argument("--lr_factor", type=float, default=0.1,
                        help="Factor to reduce learning rate by on plateau")
    parser.add_argument("--checkpoint", default="", type=str,
                        help="checkpoint name (default: none)")
    parser.add_argument("--start-epoch", default=1, type=int,
                        help="Manual epoch number (useful on restarts)")
    parser.add_argument("--clip", type=float, default=0.4,
                        help="Clipping Gradients. Default=0.4")
    parser.add_argument("--threads", type=int, default=1,
                        help=" ".join("Number of threads for data loader",
                                      "to use Default: 1"))
    parser.add_argument("--momentum", "--m", default=0.9, type=float,
                        help="Momentum, Default: 0.9")
    parser.add_argument("--weight-decay", "--wd",
                        default=1e-4, type=float,
                        help="Weight decay, Default: 1e-4")
    parser.add_argument('--pretrained', default='', type=str,
                        help='name of pretrained model (default: none)')
    #Any unknown argument will go to unparsed
    args, unparsed = parser.parse_known_args()

    #Config file modifiable parameters
    config = configparser.ConfigParser()
    config.read(os.path.join('config','main.ini'))

    print('Program started with the following options')
    helpers.print_config(config)
    print('CL arguments')
    print(args)
    main(args, config)
