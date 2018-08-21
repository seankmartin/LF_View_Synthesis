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
from torch.optim.lr_scheduler import CosineAnnealingLR

import cnn_utils
from full_model import setup_model, setup_depth_model
from depth_model import DepthModel
from data_loading import create_dataloaders
import helpers
from data_transform import undo_remap

CONTINUE_MESSAGE = "==> Would you like to continue training?"
SAVE_MESSAGE = "==> Would you like to save the model?"

def main(args, config, writer):
    best_loss = math.inf
    best_model, best_epoch = None, None
    cuda = cnn_utils.check_cuda(config)

    #Attempts to otimise - see
    #https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do
    torch.backends.cudnn.benchmark = True

    data_loaders = create_dataloaders(args, config)

    model, criterion, optimizer, lr_scheduler = setup_model(args)
    depth_model, depth_criterion, depth_optim, depth_schedule = setup_depth_model(args)

    if cuda: # GPU support
        model = model.cuda()
        depth_model = depth_model.cuda()
        #The below is only needed if loss fn has params
        #criterion = criterion.cuda()

    if args.checkpoint: # Resume from a checkpoint
        best_loss = cnn_utils.load_from_checkpoint(
                        model, optimizer, args, config)

    if args.pretrained: # Direct copy weights from another model
        cnn_utils.load_weights(model, args, config, frozen=args.frozen)

    # Perform training and testing
    print("Beginning training loop")
    for epoch in range(args.start_epoch, args.start_epoch + args.nEpochs):
        epoch_loss = train(
            model=model, depth_model=depth_model, 
            depth_optim=depth_optim, depth_criterion=depth_criterion,
            depth_schedule=depth_schedule,
            dset_loaders=data_loaders,
            optimizer=optimizer, lr_scheduler=lr_scheduler,
            criterion=criterion, epoch=epoch,
            cuda=cuda, clip=args.clip, writer=writer)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch = epoch
            best_model = copy.deepcopy(model)

        #Update the scheduler - restarting
        if lr_scheduler.last_epoch == lr_scheduler.T_max:
            for group in optimizer.param_groups:
                group['lr'] = args.lr
            lr_scheduler = CosineAnnealingLR(
                optimizer,
                T_max = lr_scheduler.T_max * 2)

        cnn_utils.log_all_layer_weights(model, writer, epoch)

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

    # Save the best model
    if save:
        cnn_utils.save_checkpoint(
            best_model, best_epoch, optimizer, best_loss,
            config['PATH']['model_dir'],
            args.tag + "_best_at{}.pth".format(best_epoch)
        )

    parent_dir = os.path.abspath(os.pardir)
    scalar_dir = os.path.join(parent_dir, "logs", args.tag)
    if not os.path.isdir(scalar_dir):
        pathlib.Path(scalar_dir).mkdir(parents=True, exist_ok=True)
    writer.export_scalars_to_json(
        os.path.join(scalar_dir, "all_scalars.json"))
    writer.close()

def train(model, depth_model, depth_optim, depth_criterion,
          depth_schedule,
          dset_loaders, optimizer, lr_scheduler,
          criterion, epoch, cuda, clip, writer):
    """
    Trains model using data_loader with the given
    optimizer, lr_scheduler, criterion and epoch
    """
    lr_scheduler.step()
    depth_schedule.step()
    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        since = time.time()
        if phase == 'train':
            model.train() # Set model to training mode
        else:
            model.eval() # Set model to evaluate mode

        running_loss = 0.0
        for iteration, batch in enumerate(dset_loaders[phase]):
            # Use this if doing cyclic learning
            # lr_scheduler.batch_step()
            sample_index = 8 * 4 + 4
            d_targets = batch['depth'].transpose_(4, 2).transpose_(3, 4)
            d_inputs = batch['depth'][:, sample_index]
            d_inputs.transpose_(3, 1).transpose_(2, 3)
            d_inputs.requires_grad_()
            d_targets.requires_grad_(False)

            if cuda:
                d_inputs = d_inputs.cuda()
                targets = targets.cuda()

            depth_out = depth_model(d_inputs)
            loss = depth_criterion(depth_out, d_targets)
            depth_optim.zero_grad()

            d_running_loss += loss.item()

            if iteration % 100 == 0:
                print("===> Epoch[{}]({}/{}): Loss: {:.5f}".format(
                        epoch, iteration, len(dset_loaders[phase]),
                        loss.item()))

            if phase == 'train':
                loss.backward()
                nn.utils.clip_grad_norm_(
                    model.parameters(), clip)
                depth_optim.step()
            
            inputs = torch.zeros((batch.shape[0],) + batch['grid_size'][0])
            targets = torch.zeros((batch.shape[0],) + batch['grid_size'][0])
            
            for i in range(batch.shape[0])
                sample = {'depth': depth_out[i], 'targets': batch['colour'][i], 'grid_size': batch['grid_size'][0]}
                inputs[i] = data_transform.transform_to_warped(sample)
                targets[i] = batch['colour'][i]
            
            inputs.requires_grad_()
            targets.requires_grad_(False)

            # forward
            if iteration == 0:
                print("Loaded " + phase + " batch in {:.0f}s".format(
                    time.time() - since))
            residuals = model(inputs)
            outputs = inputs + residuals
            outputs = torch.clamp(outputs, 0.0, 1.0)

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

            if iteration % 100 == 0:
                print("===> Epoch[{}]({}/{}): Loss: {:.5f}".format(
                    epoch, iteration, len(dset_loaders[phase]),
                    loss.item()))

                if phase == 'train':
                    if not cnn_utils.check_gradients(model):
                        print("No gradients are being computed during training")
                        exit(-1)

            if iteration == len(dset_loaders[phase]) - 1:
                desired_shape = batch['shape']
                inputs_s = undo_remap(inputs[0], desired_shape, dtype=torch.float32)
                residuals_s = undo_remap(residuals[0], desired_shape, dtype=torch.float32)
                outputs_s = undo_remap(outputs[0], desired_shape, dtype=torch.float32)
                targets_s = undo_remap(targets[0], desired_shape, dtype=torch.float32)
                input_imgs = cnn_utils.transform_lf_to_torch(inputs_s)
                residual_imgs = cnn_utils.transform_lf_to_torch(residuals_s)
                out_imgs = cnn_utils.transform_lf_to_torch(outputs_s)
                truth_imgs = cnn_utils.transform_lf_to_torch(targets_s)
                input_grid = vutils.make_grid(
                    input_imgs, nrow=8, range=(0, 1), normalize=True,
                    pad_value=1.0)
                residual_grid = vutils.make_grid(
                    residual_imgs, nrow=8, range=(-1, 1), normalize=True,
                    pad_value=1.0)
                output_grid = vutils.make_grid(
                    out_imgs, nrow=8, range=(0, 1), normalize=True,
                    pad_value=1.0)
                target_grid = vutils.make_grid(
                    truth_imgs, nrow=8, range=(0, 1), normalize=True,
                    pad_value=1.0)
                diff_grid = vutils.make_grid(
                    torch.abs(truth_imgs - out_imgs),
                    nrow=8, range=(0, 1), normalize=True,
                    pad_value=1.0)
                writer.add_image(phase + '/input', input_grid, epoch)
                writer.add_image(phase + '/residual', residual_grid, epoch)
                writer.add_image(phase + '/output', output_grid, epoch)
                writer.add_image(phase + '/target', target_grid, epoch)
                writer.add_image(phase + '/difference', diff_grid, epoch)

        d_epoch_loss = d_running_loss / len(dset_loaders[phase])
        writer.add_scalar(phase + '/loss', d_epoch_loss, epoch)
        print("Phase {} average overall loss {:.5f}".format(phase, d_epoch_loss))
        time_elapsed = time.time() - since
        print("Phase {} took {:.0f}s overall".format(phase, time_elapsed))
        
        epoch_loss = running_loss / len(dset_loaders[phase])
        writer.add_scalar(phase + '/loss', epoch_loss, epoch)
        print("Phase {} average overall loss {:.5f}".format(phase, epoch_loss))
        time_elapsed = time.time() - since
        print("Phase {} took {:.0f}s overall".format(phase, time_elapsed))

        if phase == 'val':
            print()
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
    PARSER.add_argument('--config', "--cfg", default='main.ini', type=str,
                        help="Name of config file to use")
    PARSER.add_argument('--n_feats', '--nf', default=8, type=int,
                        help="Number of features to use, default 64")
    PARSER.add_argument('--n_resblocks', '--nr', default=4, type=int,
                        help="Number of residual blocks, default 10")
    PARSER.add_argument('--res_scale', '--rs', default=1.0, type=float,
                        help="Float to scale residuals by, default 1.0")
    PARSER.add_argument('--frozen', '--f', default=True, type=bool,
                        help="Should the loaded weights be frozen?, default=True")
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
    CONFIG.read(os.path.join('config', ARGS.config))
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
