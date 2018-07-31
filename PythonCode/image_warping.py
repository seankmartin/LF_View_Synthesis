"""Image warping based on a disparity"""
import configparser
import os

import h5py
from PIL import Image
import numpy as np

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

def save_array_as_image(array, save_location):
    """Saves an array as an image at the save_location using pillow"""
    image = Image.fromarray(array)
    image.save(save_location)
    image.close()

def get_diff_image(im1, im2):
    diff = np.subtract(im1.astype(float), im2.astype(float))
    diff = abs(diff).astype(np.uint8)
    return diff

def main(config):
    hdf5_path = os.path.join(config['PATH']['output_dir'],
                             config['PATH']['hdf5_name'])
    with h5py.File(hdf5_path, mode='r', libver='latest') as hdf5_file:
        depth_grp = hdf5_file['train']['disparity']
        SNUM = 0
        depth_image = depth_grp['images'][SNUM, 27, :, :, 0]

        #Hardcoded some values for now
        colour_grp = hdf5_file['train']['colour']
        colour_image = colour_grp['images'][SNUM, 27]

        #Can later expand like 0000 if needed
        base_dir = os.path.join(config['PATH']['output_dir'], 'warped')
        get_diff = (config['DEFAULT']['should_get_diff'] == 'True')
        for i in range(8):
            for j in range(8):
                res = fw_warp_image(colour_image, depth_image,
                                    np.asarray([3, 3]), np.asarray([i, j]))
                file_name = 'Colour{}{}.png'.format(i, j)
                save_location = os.path.join(base_dir, file_name)
                save_array_as_image(res, save_location)
                if get_diff:
                    colour = colour_grp['images'][SNUM, i * 8 + j]
                    diff = get_diff_image(colour, res)
                    file_name = 'Diff{}{}.png'.format(i, j)
                    save_location = os.path.join(base_dir, file_name)
                    save_array_as_image(diff, save_location)

if __name__ == '__main__':
    CONFIG = configparser.ConfigParser()
    CONFIG.read(os.path.join('config', 'hdf5.ini'))
    DIRTOMAKE = os.path.join(CONFIG['PATH']['output_dir'], 'warped')
    if not os.path.exists(DIRTOMAKE):
        os.makedirs(DIRTOMAKE)
    main(CONFIG)
