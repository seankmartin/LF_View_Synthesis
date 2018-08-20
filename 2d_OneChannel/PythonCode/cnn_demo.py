"""Take in input, perform CNN on it, and return an output"""
import argparse
import configparser
import os
import time
import pathlib
import math

import torch
import h5py
import numpy as np
from PIL import Image

import cnn_utils
import conversions
import helpers
import data_transform
from data_transform import denormalise_lf
import image_warping
import evaluate
import welford

def get_sub_dir_for_saving(base_dir):
    """
    Returns the number of sub directories of base_dir, n, in format
    base_dir + path_separator + n
    Where n is padded on the left by zeroes to be of length four

    Example: base_dir is /home/sean/test with two sub directories
    Output: /home/sean/test/0002
    """
    num_sub_dirs = sum(os.path.isdir(os.path.join(base_dir, el))
                   for el in os.listdir(base_dir))

    sub_dir_to_save_to_name = str(num_sub_dirs)
    sub_dir_to_save_to_name = sub_dir_to_save_to_name.zfill(4)

    sub_dir_to_save_to = os.path.join(base_dir, sub_dir_to_save_to_name)
    os.mkdir(sub_dir_to_save_to)

    return sub_dir_to_save_to

def main(args, config, sample_index):
    cuda = cnn_utils.check_cuda(config)
    model = cnn_utils.load_model_and_weights(args, config)
    if cuda:
        model = model.cuda()

    model.eval()

    # Create output directory
    base_dir = os.path.join(config['PATH']['output_dir'], 'warped')
    if not os.path.isdir(base_dir):
        pathlib.Path(base_dir).mkdir(parents=True, exist_ok=True)
    save_dir = get_sub_dir_for_saving(base_dir)

    # TODO if GT is available, can get diff images
    start_time = time.time()
    file_path = os.path.join(config['PATH']['hdf5_dir'],
                                config['PATH']['hdf5_name'])
    with h5py.File(file_path, mode='r', libver='latest') as hdf5_file:
        depth_grp = hdf5_file['val']['disparity']
        SNUM = sample_index
        depth_images = torch.squeeze(torch.tensor(
            depth_grp['images'][SNUM],
            dtype=torch.float32))

        colour_grp = hdf5_file['val']['colour']
        colour_images = torch.tensor(
            colour_grp['images'][SNUM],
            dtype=torch.float32)

        sample = {'depth': depth_images,
                  'colour': colour_images,
                  'grid_size': depth_images.shape[0]}

        warped = data_transform.transform_to_warped(sample)
        warped = data_transform.stack(warped)
        im_input = warped['inputs'].unsqueeze_(0)

        if cuda:
            im_input = im_input.cuda()

        output = model(im_input)
        output += im_input
        output = torch.clamp(output, 0.0, 1.0)

        time_taken = time.time() - start_time
        print("Time taken was {:.0f}s".format(time_taken))
        grid_size = 64

        psnr_accumulator = (0, 0, 0)
        ssim_accumulator = (0, 0, 0)

        print("Saving output to", save_dir)
        if args.no_cnn:
            no_cnn_dir = os.path.join(save_dir, "no_cnn")
            cnn_dir = os.path.join(save_dir, "cnn")
            os.mkdir(cnn_dir)
            os.mkdir(no_cnn_dir)

        output = torch.squeeze(denormalise_lf(output))
        output = data_transform.torch_unstack(output)
        cpu_output = np.around(output.cpu().detach().numpy()).astype(np.uint8)

        if (not args.no_eval) or args.get_diff:
            ground_truth = np.around(
                denormalise_lf(colour_images).numpy()
                ).astype(np.uint8)
        grid_len = int(math.sqrt(grid_size))
        for i in range(grid_size):
            row, col = i // grid_len, i % grid_len

            file_name = 'Colour{}{}.png'.format(row, col)
            save_location = os.path.join(cnn_dir, file_name)
            if i == 0:
                print("Saving images of size ", cpu_output[i].shape)
            image_warping.save_array_as_image(
                cpu_output[i], save_location)
            
            if args.get_diff:
                colour = ground_truth[i]
                diff = image_warping.get_diff_image(colour, cpu_output[i])
                #diff = get_diff_image_floatint(res, colour)
                file_name = 'Diff{}{}.png'.format(row, col)
                save_location = os.path.join(cnn_dir, file_name)
                image_warping.save_array_as_image(diff, save_location)

            if not args.no_eval:
                img = ground_truth[i]
                file_name = 'GT_Colour{}{}.png'.format(row, col)
                save_location = os.path.join(save_dir, file_name)
                image_warping.save_array_as_image(img, save_location)
                psnr = evaluate.my_psnr(
                    cpu_output[i], 
                    img)
                ssim = evaluate.ssim(
                    cpu_output[i], 
                    img)
                psnr_accumulator = welford.update(psnr_accumulator, psnr)
                ssim_accumulator = welford.update(ssim_accumulator, ssim)

        psnr_mean, psnr_var, _ = welford.finalize(psnr_accumulator)
        ssim_mean, ssim_var, _ = welford.finalize(ssim_accumulator)
        print("For cnn, psnr average {:5f}, stddev {:5f}".format(
            psnr_mean, math.sqrt(psnr_var)))
        print("For cnn, ssim average {:5f}, stddev {:5f}".format(
            ssim_mean, math.sqrt(ssim_var)))

        psnr_accumulator = (0, 0, 0)
        ssim_accumulator = (0, 0, 0)

        if args.no_cnn:
            squeeze_input = torch.squeeze(denormalise_lf(im_input))
            squeeze_input = data_transform.torch_unstack(squeeze_input)
            cpu_input = np.around(
                squeeze_input.cpu().detach().numpy()).astype(np.uint8)
            for i in range(grid_size):
                row, col = i // grid_len, i % grid_len

                file_name = 'Colour{}{}.png'.format(row, col)
                save_location = os.path.join(no_cnn_dir, file_name)
                if i == 0:
                    print("Saving images of size ", cpu_input[i].shape)
                image_warping.save_array_as_image(
                    cpu_input[i], save_location)

                if args.get_diff:
                    colour = ground_truth[i]
                    diff = image_warping.get_diff_image(colour, cpu_output[i])
                    #diff = get_diff_image_floatint(res, colour)
                    file_name = 'Diff{}{}.png'.format(row, col)
                    save_location = os.path.join(no_cnn_dir, file_name)
                    image_warping.save_array_as_image(diff, save_location)

                if not args.no_eval:
                    img = ground_truth[i]
                    psnr = evaluate.my_psnr(
                        cpu_input[i], 
                        img)
                    ssim = evaluate.ssim(
                        cpu_input[i], 
                        img)
                    psnr_accumulator = welford.update(psnr_accumulator, psnr)
                    ssim_accumulator = welford.update(ssim_accumulator, ssim)

            psnr_mean, psnr_var, _ = welford.finalize(psnr_accumulator)
            ssim_mean, ssim_var, _ = welford.finalize(ssim_accumulator)
            print("For no cnn, psnr average {:5f}, stddev {:5f}".format(
                psnr_mean, math.sqrt(psnr_var)))
            print("For no cnn, ssim average {:5f}, stddev {:5f}".format(
                ssim_mean, math.sqrt(ssim_var)))

        #Ground truth possible
        """
        grid_size = 64
        grid_len = int(math.sqrt(grid_size))
        for i in range(grid_size):
            row, col = i // grid_len, i % grid_len

            file_name = 'Colour{}{}.png'.format(row, col)
            save_location = os.path.join(save_dir, file_name)
            if i == 0:
                print("Saving images of size ", colour_images[i, ...].shape)
            image_warping.save_array_as_image(
                colour_images[i, ...].numpy().astype(np.uint8), save_location)
        """

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
    PARSER.add_argument('--pretrained', default="best_model.pth", type=str,
                        help=MODEL_HELP_STR)
    PARSER.add_argument('--no_cnn', action='store_true',
                        help="output the images with and without the cnn")
    PARSER.add_argument('--no_eval', action='store_true',
                        help='do not evaluate the output')
    PARSER.add_argument('--sample', default=0, type=int,
                        help='which light field sample to use, default 0')
    PARSER.add_argument('--get_diff', action='store_true',
                        help="Should get difference images")
    #Any unknown argument will go to unparsed
    ARGS, UNPARSED = PARSER.parse_known_args()

    if len(UNPARSED) is not 0:
        print("Unrecognised command line argument passed")
        print(UNPARSED)
        exit(-1)

    #Config file modifiable parameters
    CONFIG = configparser.ConfigParser()
    CONFIG.read(os.path.join('config', 'main.ini'))

    print('Program started with the following options')
    helpers.print_config(CONFIG)
    print('Command Line arguments')
    print(ARGS)
    print()
    main(ARGS, CONFIG, ARGS.sample)
