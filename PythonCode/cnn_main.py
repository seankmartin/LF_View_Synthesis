"""
The main file to run the View synthesis CNN
Many Neural Network pipeline considerations are based on
https://github.com/twtygqyy/pytorch-vdsr
And the pytorch example files
"""
import os
import copy
import argparse
import configparser
import time
import math

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from data_loading import DatasetFromHdf5
import helpers

def main(args, config):
    #Preliminary setup
    cuda = config['Network']['cuda'] == 'True'
    cuda_device = config['Network']['gpu_id']
    if cuda:
        print("=> use gpu id: '{}'".format(cuda_device))
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id")
        print("cudnn version is", torch.backends.cudnn.version())

    #Attempts to otimise - see
    #https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do
    torch.backends.cudnn.benchmark = True

    print("Loading dataset")
    #TODO There should be a train and val here
    #this can be done with the h5
    h5_lf_vis_file_loc = os.path.join(config['PATH']['hdf5_dir'],
                                      config['PATH']['hdf5_name'])
    dataset = DatasetFromHdf5(h5_lf_vis_file_loc)
    data_loader = DataLoader(dataset=dataset, num_workers=args.threads,
                             batch_size=int(config['NETWORK']['batch_size']),
                             shuffle=True)

    # Load the appropriate model with gpu support
    print("Building model")
    model = None
    criterion = nn.MSELoss(size_average=False)
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay,
			  nesterov=True)
    lr_scheduler = ReduceLROnPlateau(
        optimizer, 'min', factor=args.lr_factor,
        patience=3, threshold=1e-3,
        threshold_mode='rel', verbose=True)
    if cuda:
        model = model.cuda()
        #The below is only needed if loss fn has params
        #criterion = criterion.cuda()

    # optionally resume from a checkpoint
    if args.checkpoint:
        resume_location = os.path.join(
            config['PATH']['checkpoint_dir'],
            args.checkpoint)
        if os.path.isfile(resume_location):
            print("=> loading checkpoint '{}'".format(resume_location))
            checkpoint = torch.load(resume_location)
            args.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["model"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(resume_location))

    # optionally copy weights from a checkpoint
    if args.pretrained:
        weights_location = os.path.join(
            config['PATH']['model_dir'],
            args.pretrained)
        if os.path.isfile(weights_location):
            print("=> loading model '{}'".format(weights_location))
            weights = torch.load(weights_location)
            model.load_state_dict(weights['model'].state_dict())
        else:
            print("=> no model found at '{}'".format(weights_location))

    # Perform training and testing
    best_loss = math.inf
    for epoch in range(args.start_epoch, args.nEpochs + 1):
        epoch_loss = train(model=model, dset_loaders=data_loader,
                           optimizer=optimizer, lr_scheduler=lr_scheduler,
                           criterion=criterion, epoch=epoch, 
                           cuda=cuda, clip=args.clip)
        # deep copy the model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model = copy.deepcopy(model)
            best_epoch = epoch
        if epoch % 5 == 0:
            save_checkpoint(
                model, epoch,
                config['PATH']['model_dir'], args.tag + "{}.pth".format(epoch))
    save_checkpoint(
        best_model, best_epoch,
        config['PATH']['model_dir'],
        args.tag + "_best_at{}.pth".format(best_epoch))

    # Clean up
    dataset.close_h5()

def train(model, dset_loaders, optimizer, lr_scheduler,
          criterion, epoch, cuda, clip):
    """
    Trains model using data_loader with the given
    optimizer, lr_scheduler, criterion and epoch
    """
    since = time.time()

    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()  # Set model to evaluate mode

        running_loss = 0.0
        for iteration, batch in enumerate(dset_loaders[phase]):
            inputs, targets = batch
            inputs.requires_grad_()
            targets.requires_grad_(False)

            if cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()

            # forward
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()

            # backward + optimize only if in training phase
            if phase == 'train':
                loss.backward()
                nn.utils.clip_grad_norm(
                    model.parameters(), clip)
                optimizer.step()

            # statistics
            running_loss += loss.data[0]

            if iteration%100 == 0:
                print("===> Epoch[{}]({}/{}): Loss: {:.10f}".format(
                    epoch, iteration, len(dset_loaders[phase]),
                    loss.data[0]))

        epoch_loss = running_loss / len(dset_loaders[phase])
        print("Phase {} overall loss {:.4f}".format(phase, epoch_loss))
        time_elapsed = time.time() - since
        print("Phase {} took {} overall".format(phase, time_elapsed))

        if phase == 'val':
            print('\n', end='')
            lr_scheduler.step(epoch_loss)
            return epoch_loss

def save_checkpoint(model, epoch, save_dir, name):
    """Saves model params and epoch number at save_dir/name"""
    model_out_path = os.path.join(save_dir, name)
    state = {"epoch": epoch, "model": model}
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch.save(state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))

if __name__ == '__main__':
    #Command line modifiable parameters
    #See https://github.com/twtygqyy/pytorch-vdsr/blob/master/main_vdsr.py
    #For the source of some of these arguments
    THREADS_HELP = " ".join(("Number of threads for data loader",
                             "to use Default: 1"))
    PARSER = argparse.ArgumentParser(
        description='Process modifiable parameters from command line')
    PARSER.add_argument('--base_dir', type=str, default="",
                        help='base directory for the data')
    PARSER.add_argument("--nEpochs", type=int, default=50,
                        help="Number of epochs to train for")
    PARSER.add_argument("--lr", type=float, default=0.1,
                        help="Learning Rate. Default=0.1")
    PARSER.add_argument("--lr_factor", type=float, default=0.1,
                        help="Factor to reduce learning rate by on plateau")
    PARSER.add_argument("--checkpoint", default="", type=str,
                        help="checkpoint name (default: none)")
    PARSER.add_argument("--start-epoch", default=1, type=int,
                        help="Manual epoch number (useful on restarts)")
    PARSER.add_argument("--clip", type=float, default=0.4,
                        help="Clipping Gradients. Default=0.4")
    PARSER.add_argument("--threads", type=int, default=1,
                        help=THREADS_HELP)
    PARSER.add_argument("--momentum", "--m", default=0.9, type=float,
                        help="Momentum, Default: 0.9")
    PARSER.add_argument("--weight-decay", "--wd",
                        default=1e-4, type=float,
                        help="Weight decay, Default: 1e-4")
    PARSER.add_argument('--pretrained', default='', type=str,
                        help='name of pretrained model (default: none)')
    PARSER.add_argument('--tag', default=None, type=str,
                        help='Unique identifier for a model training run')
    #Any unknown argument will go to unparsed
    ARGS, UNPARSED = PARSER.parse_known_args()
    if ARGS.tag is None:
        print("Please enter a --tag flag through cmd when running")
        exit(-1)

    #Config file modifiable parameters
    CONFIG = configparser.ConfigParser()
    CONFIG.read(os.path.join('config', 'main.ini'))

    print('Program started with the following options')
    helpers.print_config(CONFIG)
    print('CL arguments')
    print(ARGS)
    main(ARGS, CONFIG)
