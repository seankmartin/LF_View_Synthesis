"""Image warping based on a disparity"""
import configparser
import os
from enum import Enum

import h5py
from PIL import Image
import numpy as np
from skimage.transform import warp
import math

import welford
import evaluate

class WARP_TYPE(Enum):
    FW = 1
    SK = 2
    SLOW = 3

def shift_disp(xy, disp, distance, dtype):
    #Repeat the elements of the disparity_map to match the distance
    size_x, size_y = disp.shape[0:2]
    
    #Needs to be tranposed to match expected cols, rows
    repeated = np.repeat(disp.T, 2, -1).reshape((size_x * size_y, 2))

    #Convert to desired dtype
    result = (repeated * distance).astype(dtype)
    return xy - result

def sk_warp(
    ref_view, disparity_map, ref_pos, novel_pos, 
    dtype=np.float32, blank=0, preserve_range=False):
    """
    Uses skimage to perform backward warping:
    
    Keyword arguments:
    ref_view -- colour image data at the reference position
    disparity_map -- a disparity map at the reference position
    ref_pos -- the grid co-ordinates of the ref_view
    novel_pos -- the target grid position for the novel view
    dtype -- data type to consider disparity as
    blank -- value to use at positions not seen in reference view
    preserve_range -- Keep the data in range 0, 255 or convert to 0 1
    """
    distance = ref_pos - novel_pos
    
    novel_view = warp(
        image=ref_view, inverse_map=shift_disp, 
        map_args={
            "disp": disparity_map, "distance": np.flipud(distance), 
            "dtype": dtype},
        cval=blank, preserve_range=preserve_range, order=1
        )
    if preserve_range:
        novel_view = novel_view.astype(np.uint8)
    return novel_view

def valid_pixel(pixel, img_size):
    """Returns true if the pixel co-ordinate lies inside the image grid"""
    size_x, size_y = img_size
    valid = (((pixel[0] > -1) and (pixel[0] < size_x)) and
             ((pixel[1] > -1) and (pixel[1] < size_y)))
    return valid

def fw_warp_image(
    ref_view, disparity_map, ref_pos, novel_pos, 
    dtype=np.uint8, blank=0):
    """
    Returns a forward warped novel from an input image and disparity_map
    For each pixel position in the reference view, shift it by the disparity,
    and assign the value in the reference at that new pixel position to the
    novel view.

    Keyword arguments:
    ref_view -- colour image data at the reference position
    disparity_map -- a disparity map at the reference position
    ref_pos -- the grid co-ordinates of the ref_view
    novel_pos -- the target grid position for the novel view
    """
    size_x, size_y = ref_view.shape[0:2]
    distance = ref_pos - novel_pos

    #Initialise an array of blanks
    novel_view = np.full(ref_view.shape, blank, dtype=dtype)

    #Create an array of pixel positions
    grid = np.meshgrid(np.arange(size_x), np.arange(size_y), indexing='ij')
    stacked = np.stack(grid, 2)
    pixels = stacked.reshape(-1, 2)

    #Repeat the elements of the disparity_map to match the distance
    repeated = np.repeat(disparity_map, 2, -1).reshape((size_x * size_y, 2))

    #Round to the nearest integer value
    result = (repeated * distance).astype(int)
    novel_pixels = pixels + result
    
    #Move the pixels from the reference view to the novel view
    for novel_coord, ref_coord in zip(novel_pixels, pixels):
        if valid_pixel(novel_coord, ref_view.shape[0:2]):
            novel_view[novel_coord[0], novel_coord[1]] = (
                ref_view[ref_coord[0], ref_coord[1]])

    return novel_view

def slow_fw_warp_image(ref_view, disparity_map, ref_pos, novel_pos):
    """
    Returns a forward warped novel from an input image and disparity_map
    For each pixel position in the reference view, shift it by the disparity,
    and assign the value in the reference at that new pixel position to the
    novel view.
    Has a very large for loop, performance is much slower than
    fw_warp_image

    Keyword arguments:
    ref_view -- colour image data at the reference position
    disparity_map -- a disparity map at the reference position
    ref_pos -- the grid co-ordinates of the ref_view
    novel_pos -- the target grid position for the novel view
    """
    size_x, size_y = ref_view.shape[0:2]
    distance = ref_pos - novel_pos

    novel_view = np.zeros(ref_view.shape, dtype=np.uint8)
    for x in range(size_x):
        for y in range(size_y):
            res = np.repeat(disparity_map[x, y], 2, -1) * distance
            new_pixel = ((x, y) + res).astype(int)
            if valid_pixel(new_pixel, (size_x, size_y)):
                novel_view[new_pixel[0], new_pixel[1]] = ref_view[x, y]
    return novel_view

def save_array_as_image(array, save_location):
    """Saves an array as an image at the save_location using pillow"""
    image = Image.fromarray(array)
    image.save(save_location)
    image.close()

def get_diff_image(im1, im2):
    diff = np.subtract(im1.astype(float), im2.astype(float))
    diff = abs(diff).astype(np.uint8)
    return diff

def get_diff_image_floatint(im1_float, im2_int):
    diff = np.subtract(im1_float, im2_int.astype(float) / 255.0)
    diff = abs(diff).astype(np.uint8)
    return diff

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

def main(config):
    hdf5_path = os.path.join(config['PATH']['output_dir'],
                             config['PATH']['hdf5_name'])
    warp_type = WARP_TYPE.SK
    with h5py.File(hdf5_path, mode='r', libver='latest') as hdf5_file:
        grid_size = 64
        grid_one_way = 8
        sample_index = grid_size // 2 + (grid_one_way // 2)
        depth_grp = hdf5_file['train']['disparity']
        SNUM = 2
        depth_image = depth_grp['images'][SNUM, sample_index, :, :, 0]

        #Hardcoded some values for now
        colour_grp = hdf5_file['train']['colour']
        colour_image = colour_grp['images'][SNUM, sample_index]

        #Can later expand like 0000 if needed
        base_dir = os.path.join(config['PATH']['output_dir'], 'warped')
        get_diff = (config['DEFAULT']['should_get_diff'] == 'True')
        save_dir = get_sub_dir_for_saving(base_dir)
        print("Saving images to {}".format(save_dir))

        psnr_accumulator = (0, 0, 0)
        ssim_accumulator = (0, 0, 0)
        for i in range(8):
            for j in range(8):
                if warp_type is WARP_TYPE.FW:
                    res = fw_warp_image(colour_image, depth_image,
                                        np.asarray([4, 4]), np.asarray([i, j]))
                elif warp_type is WARP_TYPE.SK:
                    res = sk_warp(
                        colour_image, depth_image,
                        np.asarray([4, 4]), np.asarray([i, j]),
                        preserve_range=True
                    )
                elif warp_type is WARP_TYPE.SLOW:
                    res = slow_fw_warp_image(
                        colour_image, depth_image,
                        np.asarray([4, 4]), np.asarray([i, j])
                    )
                file_name = 'Warped_Colour{}{}.png'.format(i, j)
                save_location = os.path.join(save_dir, file_name)
                save_array_as_image(res, save_location)
                idx = i * 8 + j
                file_name = 'GT_Colour{}{}.png'.format(i, j)
                save_location = os.path.join(save_dir, file_name)
                save_array_as_image(
                    colour_grp['images'][SNUM][idx], save_location)
                if get_diff:
                    colour = colour_grp['images'][SNUM, i * 8 + j]
                    diff = get_diff_image(colour, res)
                    #diff = get_diff_image_floatint(res, colour)
                    file_name = 'Diff{}{}.png'.format(i, j)
                    save_location = os.path.join(save_dir, file_name)
                    save_array_as_image(diff, save_location)
                psnr = evaluate.my_psnr(
                    res, 
                    colour_grp['images'][SNUM, i * 8 + j])
                ssim = evaluate.ssim(
                    res, 
                    colour_grp['images'][SNUM, i * 8 + j])
                psnr_accumulator = welford.update(psnr_accumulator, psnr)
                ssim_accumulator = welford.update(ssim_accumulator, ssim)

        psnr_mean, psnr_var, _ = welford.finalize(psnr_accumulator)
        ssim_mean, ssim_var, _ = welford.finalize(ssim_accumulator)
        print("psnr average {:5f}, stddev {:5f}".format(
            psnr_mean, math.sqrt(psnr_var)))
        print("ssim average {:5f}, stddev {:5f}".format(
            ssim_mean, math.sqrt(ssim_var)))

if __name__ == '__main__':
    CONFIG = configparser.ConfigParser()
    CONFIG.read(os.path.join('config', 'hdf5.ini'))
    DIRTOMAKE = os.path.join(CONFIG['PATH']['output_dir'], 'warped')
    if not os.path.exists(DIRTOMAKE):
        os.makedirs(DIRTOMAKE)
    main(CONFIG)
