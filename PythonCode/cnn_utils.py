"""Handles functions such as loading from checkpoints"""
import os

import torch

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

def print_mem_usage():
    """Prints torch gpu memeory usage"""
    print("Using {} / {} tensor memory".format(
        torch.cuda.memory_allocated(),
        torch.cuda.max_memory_allocated()
    ))
    print("Using {} / {} cached memory".format(
        torch.cuda.memory_cached(),
        torch.cuda.max_memory_cached()
    ))