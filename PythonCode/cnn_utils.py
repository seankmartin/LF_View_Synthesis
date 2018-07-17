"""Handles functions such as loading from checkpoints"""
import os

import torch

import data_transform
from data_loading import TrainFromHdf5, ValFromHdf5
from torch.utils.data import DataLoader

def check_cuda(config):
    """Checks cuda settings from config - Returns true if cuda available"""
    cuda = config['NETWORK']['cuda'] == 'True'
    cuda_device = config['NETWORK']['gpu_id']
    if cuda:
        print("=> using gpu id: '{}'".format(cuda_device))
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id")
        print("=> cudnn version is", torch.backends.cudnn.version())
    return cuda

def save_checkpoint(model, epoch, save_dir, name):
    """Saves model params and epoch number at save_dir/name"""
    model_out_path = os.path.join(save_dir, name)
    state = {"epoch": epoch, "model": model}
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch.save(state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))

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

def create_dataloaders(hdf_file, args, config):
    """Creates a train and val dataloader from a h5file and a config file"""
    print("Loading dataset")
    train_set = TrainFromHdf5(
        hdf_file=hdf_file,
        patch_size=int(config['NETWORK']['patch_size']),
        num_crops=int(config['NETWORK']['num_crops']),
        transform=data_transform.transform_to_warped)
    val_set = ValFromHdf5(
        hdf_file=hdf_file, 
        transform=data_transform.transform_to_warped)

    batch_size = {'train': int(config['NETWORK']['batch_size']), 'val': 1}
    data_loaders = {}
    for name, dset in (('train', train_set), ('val', val_set)):
        data_loaders[name] = DataLoader(
            dataset=dset, num_workers=args.threads,
            batch_size=batch_size[name],
            shuffle=True)

    return data_loaders
