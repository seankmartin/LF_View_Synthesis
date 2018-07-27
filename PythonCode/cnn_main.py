""" The main file to run the View synthesis CNN
Many Neural Network pipeline considerations are based on
https://github.com/twtygqyy/pytorch-vdsr
And the pytorch example files """
import argparse
import configparser
import copy
import math
import os
import time
import pathlib

import torch
import torch.nn as nn
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

import cnn_utils
from full_model import setup_model
from data_loading import create_dataloaders
import helpers

CONTINUE_MESSAGE = "==> Would you like to continue training?"
SAVE_MESSAGE = "==> Would you like to save the model?"

def main(args, config, writer):
    best_loss = math.inf
    best_epoch = None
    cuda = cnn_utils.check_cuda(config)

    #Attempts to otimise - see
    #https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do
    torch.backends.cudnn.benchmark = True

    data_loaders = create_dataloaders(args, config)

    model, criterion, optimizer, lr_scheduler = setup_model(args)
    if cuda: # GPU support
        model = model.cuda()
        #The below is only needed if loss fn has params
        #criterion = criterion.cuda()

    if args.checkpoint: # Resume from a checkpoint
        best_loss = cnn_utils.load_from_checkpoint(
                        model, optimizer, args, config)

    if args.pretrained: # Direct copy weights from another model
        cnn_utils.load_weights(model, args, config)

    # Perform training and testing
    print("Beginning training loop")
    for epoch in range(args.start_epoch, args.start_epoch + args.nEpochs):
        epoch_loss = train(
            model=model, dset_loaders=data_loaders,
            optimizer=optimizer, lr_scheduler=lr_scheduler,
            criterion=criterion, epoch=epoch,
            cuda=cuda, clip=args.clip, writer=writer)    

        if epoch == args.start_epoch:
            avg_model = copy.deepcopy(model)
        else:
            cnn_utils.merge_weights(avg_model, model, 0.6)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch = epoch

        layer_weights = model.state_dict()['first.conv.weight']
        writer.add_histogram(
            "Layer 0 weight", layer_weights, epoch, 'auto')

        if epoch % 5 == 0 and epoch != 0:
            cnn_utils.save_checkpoint(
                model, epoch, optimizer, best_loss,
                config['PATH']['checkpoint_dir'],
                args.tag + "{}.pth".format(epoch))

        if args.prompt:
            if not helpers.prompt_user(CONTINUE_MESSAGE):
                print("Ending training")
                break

    print("Best loss was {:.5f} at epoch {}".format(
        best_loss, best_epoch
    ))

    save = True
    if args.prompt:
        if not helpers.prompt_user(SAVE_MESSAGE):
            print("Not saving the model")
            save = False

    # Save the average model
    if save:
        cnn_utils.save_checkpoint(
            avg_model, epoch, optimizer, best_loss,
            config['PATH']['model_dir'],
            args.tag + "_avg_at{}.pth".format(epoch))

    parent_dir = os.path.abspath(os.pardir)
    scalar_dir = os.path.join(parent_dir, "logs", args.tag)
    if not os.path.isdir(scalar_dir):
        pathlib.Path(scalar_dir).mkdir(parents=True, exist_ok=True)
    writer.export_scalars_to_json(
        os.path.join(scalar_dir, "all_scalars.json"))
    writer.close()

def train(model, dset_loaders, optimizer, lr_scheduler,
          criterion, epoch, cuda, clip, writer):
    """
    Trains model using data_loader with the given
    optimizer, lr_scheduler, criterion and epoch
    """

    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        since = time.time()
        if phase == 'train':
            model.train() # Set model to training mode
        else:
            model.eval() # Set model to evaluate mode

        running_loss = 0.0
        for iteration, batch in enumerate(dset_loaders[phase]):
            targets = batch['targets']
            inputs = batch['inputs']
            inputs.requires_grad_()
            targets.requires_grad_(False)

            if cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()

            # forward
            if iteration == 0:
                print("Loaded " + phase + " batch in {:.0f}s".format(
                    time.time() - since))
            residuals = model(inputs)
            outputs = inputs + residuals

            loss = criterion(outputs, targets)
            optimizer.zero_grad()

            # backward + optimize only if in training phase
            if phase == 'train':
                loss.backward()
                nn.utils.clip_grad_norm_(
                    model.parameters(), clip)
                optimizer.step()

            # statistics
            running_loss += loss.item()

            if iteration == 0 and cuda:
                cnn_utils.print_mem_usage()

            if iteration%100 == 0:
                print("===> Epoch[{}]({}/{}): Loss: {:.5f}".format(
                    epoch, iteration, len(dset_loaders[phase]),
                    loss.item()))
                input_imgs = inputs[0, ...].transpose(1, 3)
                residual_imgs = residuals[0, ...].transpose(1, 3)
                out_imgs = outputs[0, ...].transpose(1, 3)
                truth_imgs = targets[0, ...].transpose(1, 3)
                input_grid = vutils.make_grid(
                    input_imgs, nrow=8, range=(-1, 1), normalize=True,
                    pad_value=1.0)
                residual_grid = vutils.make_grid(
                    residual_imgs, nrow=8, range=(-1, 1), normalize=True,
                    pad_value=1.0)   
                output_grid = vutils.make_grid(
                    out_imgs, nrow=8, range=(-1, 1), normalize=True,
                    pad_value=1.0)
                target_grid = vutils.make_grid(
                    truth_imgs, nrow=8, range=(-1, 1), normalize=True,
                    pad_value=1.0)
                writer.add_image(phase + '/input', input_grid, epoch)
                writer.add_image(phase + '/residual', residual_grid, epoch)
                writer.add_image(phase + '/output', output_grid, epoch)
                writer.add_image(phase + '/target', target_grid, epoch)

        epoch_loss = running_loss / len(dset_loaders[phase])
        writer.add_scalar(phase + '/loss', epoch_loss, epoch)
        print("Phase {} average overall loss {:.5f}".format(phase, epoch_loss))
        time_elapsed = time.time() - since
        print("Phase {} took {:.0f}s overall".format(phase, time_elapsed))

        if phase == 'val':
            print()
            lr_scheduler.step(epoch_loss)
            for idx, param_group in enumerate(optimizer.param_groups):
                writer.add_scalar(
                    'learning_rate', param_group['lr'], epoch)
            return epoch_loss

if __name__ == '__main__':
    #Command line modifiable parameters
    #See https://github.com/twtygqyy/pytorch-vdsr/blob/master/main_vdsr.py
    #For the source of some of these arguments
    PARSER = argparse.ArgumentParser(
        description='Process modifiable parameters from command line')
    PARSER.add_argument("--nEpochs", "--n", type=int, default=50,
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
    PARSER.add_argument("--momentum", "--m", default=0.9, type=float,
                        help="Momentum, Default: 0.9")
    PARSER.add_argument("--weight-decay", "--wd",
                        default=1e-4, type=float,
                        help="Weight decay, Default: 1e-4")
    PARSER.add_argument('--pretrained', default='', type=str,
                        help='name of pretrained model (default: none)')
    PARSER.add_argument('--prompt', action='store_true',
                        help='Prompt at the end of every epoch to continue')
    PARSER.add_argument('--tag', default=None, type=str,
                        help='Unique identifier for a model. REQUIRED')
    #Any unknown argument will go to unparsed
    ARGS, UNPARSED = PARSER.parse_known_args()
    if ARGS.tag is None:
        print("Please enter a --tag flag through cmd when running")
        exit(-1)

    if len(UNPARSED) is not 0:
        print("Unrecognised command line argument passed")
        print(UNPARSED)
        exit(-1)
    #Config file modifiable parameters
    CONFIG = configparser.ConfigParser()
    CONFIG.read(os.path.join('config', 'main.ini'))
    from datetime import datetime
    TBOARD_LOC = os.path.join(
        CONFIG['PATH']['tboard'],
        ARGS.tag + "_" + datetime.now().strftime('%b%d_%H-%M-%S'))
    WRITER = SummaryWriter(log_dir=TBOARD_LOC)

    print('Program started with the following options')
    helpers.print_config(CONFIG)
    print('Command Line arguments')
    print(ARGS)
    print()
    main(ARGS, CONFIG, WRITER)
