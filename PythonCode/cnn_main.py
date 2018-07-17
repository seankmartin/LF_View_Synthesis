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

from data_loading import TrainFromHdf5, ValFromHdf5
import image_warping
from model_3d import C3D
import helpers

def main(args, config):
    cuda = check_cuda(config)

    #Attempts to otimise - see
    #https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do
    torch.backends.cudnn.benchmark = True

    file_path = os.path.join(config['PATH']['hdf5_dir'],
                             config['PATH']['hdf5_name'])
    with h5py.File(file_path, mode='r', libver='latest', swmr=True) as h5_file:
        grid_size = {'train': h5_file['train']['colour'].attrs['shape'][1],
                     'val': h5_file['val']['colour'].attrs['shape'][1]}
        data_loaders = create_dataloaders(h5_file, args, config)

        model, criterion, optimizer, lr_scheduler = setup_model(args)
        if cuda: # GPU support
            model = model.cuda()
            #The below is only needed if loss fn has params
            #criterion = criterion.cuda()

        if args.checkpoint: # Resume from a checkpoint
            load_from_checkpoint(model, args, config)

        if args.pretrained: # Direct copy weights from another model
            load_weights(model, args, config)

        # Perform training and testing
        best_loss = math.inf
        for epoch in range(args.start_epoch, args.nEpochs):
            epoch_loss = train(
                model=model, dset_loaders=data_loaders,
                optimizer=optimizer, lr_scheduler=lr_scheduler,
                criterion=criterion, epoch=epoch,
                cuda=cuda, clip=args.clip, grid_size=grid_size)

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model = copy.deepcopy(model)
                best_epoch = epoch

            if epoch % 5 == 0 and epoch != 0:
                save_checkpoint(
                    model, epoch,
                    config['PATH']['model_dir'],
                    args.tag + "{}.pth".format(epoch))

        # Save the best model
        save_checkpoint(
            best_model, best_epoch,
            config['PATH']['model_dir'],
            args.tag + "_best_at{}.pth".format(best_epoch))

def load_from_checkpoint(model, args, config):
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

def load_weights(model, args, config):
    weights_location = os.path.join(
        config['PATH']['model_dir'],
        args.pretrained)
    if os.path.isfile(weights_location):
        print("=> loading model '{}'".format(weights_location))
        weights = torch.load(weights_location)
        model.load_state_dict(weights['model'].state_dict())
    else:
        print("=> no model found at '{}'".format(weights_location))

def train(model, dset_loaders, optimizer, lr_scheduler,
          criterion, epoch, cuda, clip, grid_size):
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
            targets = batch['colour']
            disparity = batch['depth']
            warped_images = disparity_based_rendering(
                disparity.numpy(), targets.numpy(), grid_size[phase])
            # TODO add the disparity maps to the input
            # Unqueeze is only required for single size batches - need to fix
            inputs = torch.from_numpy(warped_images).float().unsqueeze_(0)
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
            running_loss += loss.item()

            if iteration%100 == 0:
                print("===> Epoch[{}]({}/{}): Loss: {:.10f}".format(
                    epoch, iteration, len(dset_loaders[phase]),
                    loss.item()))

        epoch_loss = running_loss / len(dset_loaders[phase])
        print("Phase {} overall loss {:.4f}".format(phase, epoch_loss))
        time_elapsed = time.time() - since
        print("Phase {} took {} overall".format(phase, time_elapsed))

        if phase == 'val':
            print()
            lr_scheduler.step(epoch_loss)
            return epoch_loss

#TODO can later fix to work on batches of size greater than 1
def disparity_based_rendering(disparities, views, grid_size):
    """Returns a list of warped images using the input views and disparites"""
     # Alternatively, grid_one_way - 1 can be used below
    shape = (grid_size,) + views.shape[2:]
    warped_images = np.empty(
        shape=shape, dtype=np.uint8)
    grid_one_way = int(math.sqrt(grid_size))
    for i in range(grid_one_way):
        for j in range(grid_one_way):
            res = image_warping.fw_warp_image(
                views[0, grid_size // 2 + (grid_one_way // 2), ...],
                disparities[0, grid_size // 2 + (grid_one_way // 2), ...],
                np.asarray([grid_one_way // 2, grid_one_way // 2]),
                np.asarray([i, j])
            )
            np.insert(warped_images, i * grid_one_way + j, res, axis=0)
    return warped_images

def check_cuda(config):
    cuda = config['NETWORK']['cuda'] == 'True'
    cuda_device = config['NETWORK']['gpu_id']
    if cuda:
        print("=> using gpu id: '{}'".format(cuda_device))
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id")
        print("=> cudnn version is", torch.backends.cudnn.version())
    return cuda

def create_dataloaders(hdf_file, args, config):
    """Creates a train and val dataloader from a h5file and a config file"""
    print("Loading dataset")
    train_set = TrainFromHdf5(
        hdf_file=hdf_file,
        patch_size=int(config['NETWORK']['patch_size']),
        num_crops=int(config['NETWORK']['num_crops']),
        transform=None)
    val_set = ValFromHdf5(hdf_file=hdf_file, transform=None)

    data_loaders = {}
    for name, dset in (('train', train_set), ('val', val_set)):
        data_loaders[name] = DataLoader(
            dataset=dset, num_workers=args.threads,
            batch_size=int(config['NETWORK']['batch_size']),
            shuffle=True)

    return data_loaders

def setup_model(args):
    """Returns a tuple of the model, criterion, optimizer and lr_scheduler"""
    print("Building model")
    model = C3D(inchannels=64, outchannels=64)
    criterion = nn.MSELoss(size_average=False)
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True)
    lr_scheduler = ReduceLROnPlateau(
        optimizer,
        'min', factor=args.lr_factor,
        patience=3, threshold=1e-3,
        threshold_mode='rel', verbose=True)

    return (model, criterion, optimizer, lr_scheduler)

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
                             "to use. Default: 1"))
    PARSER = argparse.ArgumentParser(
        description='Process modifiable parameters from command line')
    PARSER.add_argument("--nEpochs", type=int, default=50,
                        help="Number of epochs to train for")
    PARSER.add_argument("--lr", type=float, default=0.1,
                        help="Learning Rate. Default=0.1")
    PARSER.add_argument("--lr_factor", type=float, default=0.1,
                        help="Factor to reduce learning rate by. Default=0.1")
    PARSER.add_argument("--checkpoint", default="", type=str,
                        help="checkpoint name (default: none)")
    PARSER.add_argument("--start-epoch", default=0, type=int,
                        help="Manual epoch number, useful restarts. Default=0")
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
                        help='Unique identifier for a model. REQUIRED')
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
    print()
    main(ARGS, CONFIG)
