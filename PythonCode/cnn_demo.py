"""Take in input, perform CNN on it, and return an output"""
import argparse
import configparser
import os
import time

import torch
import h5py
from PIL import Image

import cnn_utils
import conversions
import helpers
import data_transform
import image_warping

def main(args, config):
    cuda = cnn_utils.check_cuda(config)

    #Load tensors to GPU with specified ID
    model_loc = os.path.join(
        config['PATH']['model_dir'],
        args.model
    )
    if cuda:
        GPU_ID = int(config['NETWORK']['gpu_id'])
        model = torch.load(
            model_loc, 
            map_location=lambda storage, loc: storage.cuda(GPU_ID))
    else:
        model = torch.load(model_loc)
    
    # TODO if GT is available, can get diff images
    start_time = time.time()
    if not args.no_hdf5:
        file_path = os.path.join(config['PATH']['hdf5_dir'],
                                  config['PATH']['hdf5_name'])
        with h5py.File(file_path, mode='r', libver='latest') as hdf5_file:
            depth_grp = hdf5_file['val']['disparity']
            SNUM = 3
            depth_images = depth_grp['images'][SNUM, ...]

            colour_grp = hdf5_file['val']['colour']
            colour_images = colour_grp['images'][SNUM, ...]

            base_dir = os.path.join(config['PATH']['output_dir'], 'warped')

            warped = data_transform.disparity_based_rendering(
                depth_images,
                colour_images,
                depth_images.shape[0])
            im_input = torch.from_numpy(warped).float()

            if cuda:
                im_input = im_input.cuda()

            output = model(im_input)

    else:
        #Expect folder to have format, depth0.png, colour0.png ...
        INPUT_IMAGES = 1
        for i in range(INPUT_IMAGES):
            end_str = str(i) + '.png'
            depth_loc = os.path.join(
                config['PATH']['image_dir'],
                'depth' + end_str
            )
            colour_loc = os.path.join(
                config['PATH']['image_dir'],
                'colour' + end_str
            )
            depth = Image.open(depth_loc)
            depth.load()
            #Convert to disparity - needs metadata
            #warp it as in hdf5
            #combine with other warped
            #do model

    time_taken = time.time() - start_time
    print("Time taken was {:.0f}s".format(time_taken))
    grid_size = 64
    base_dir = os.path.join(
        config['PATH']['output_dir'],
        'warped')
    for i in range(grid_size):
        file_name = 'Colour{}.png'.format(i)
        save_location = os.path.join(base_dir, file_name)
        image_warping.save_array_as_image(
            output[i].numpy(), save_location)

if __name__ == '__main__':
    MODEL_HELP_STR = ' '.join((
        'Model name to load from config model dir',
        'Default value is best_model.pth'))
    HDF_HELP_STR = ' '.join((
        'Should a hdf5 file be used for input?',
        'If yes, config specified hdf5 file is used',
        'Otherwise image_dir is used'))
    PARSER = argparse.ArgumentParser(
        description='Process modifiable parameters from command line')
    PARSER.add_argument('--model', default=None, type=str,
                        help=MODEL_HELP_STR)
    PARSER.add_argument('--no_hdf5', action='store_true',
                        help=HDF_HELP_STR)
    #Any unknown argument will go to unparsed
    ARGS, UNPARSED = PARSER.parse_known_args()

    #Config file modifiable parameters
    CONFIG = configparser.ConfigParser()
    CONFIG.read(os.path.join('config', 'main.ini'))

    print('Program started with the following options')
    helpers.print_config(CONFIG)
    print('Command Line arguments')
    print(ARGS)
    print()
    main(ARGS, CONFIG)
