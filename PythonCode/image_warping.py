import configparser
import os

import h5py
from PIL import Image
import numpy as np

import conversions as cs

def valid_pixel(pixel, img_size):
    """Returns true if the pixel co-ordinate lies inside the image grid"""
    size_x, size_y = img_size
    valid = ( ((pixel[0] > -1) and (pixel[0] < size_x)) and
              ((pixel[1] > -1) and (pixel[1] < size_y)) )
    return valid

#TODO consider keeping pixels around so don't have to make it multiple times
def fw_warp_image(ref_view, disparity_map, ref_pos, novel_pos):
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

    #Initialise an array of zeroes
    novel_view = np.zeros(ref_view.shape, dtype = np.uint8)

    #Create an array of pixel positions
    grid = np.meshgrid(np.arange(size_x), np.arange(size_y), indexing = 'ij')
    stacked = np.stack(grid, 2)
    pixels = stacked.reshape(-1, 2)

    #Repeat the elements of the disparity_map to match the distance
    repeated = np.repeat(disparity_map, 2, -1).reshape((size_x * size_y, 2))

    #Round to the nearest integer value
    result = (repeated * distance).astype(int)
    novel_pixels = pixels + result

    #Move the pixels from the reference view to the novel view
    for x, y in zip(novel_pixels, pixels):
        if valid_pixel(x, ref_view.shape[0:2]):
            novel_view[x[0], x[1]] = ref_view[y[0], y[1]]

    return novel_view

def slow_fw_warp_image(ref_view, disparity_map, ref_pos, novel_pos):
    """
    Returns a forward warped novel from an input image and disparity_map
    For each pixel position in the reference view, shift it by the disparity,
    and assign the value in the reference at that new pixel position to the
    novel view.
    Has a very large for loop, performance should be compared with
    fw_warp_image

    Keyword arguments:
    ref_view -- colour image data at the reference position
    disparity_map -- a disparity map at the reference position
    ref_pos -- the grid co-ordinates of the ref_view
    novel_pos -- the target grid position for the novel view
    """
    size_x, size_y = ref_view.shape[0:2]
    distance = ref_pos - novel_pos

    novel_view = np.zeros(ref_view.shape, dtype = np.uint8)
    for x in range(size_x):
        for y in range(size_y):
            res = np.repeat(disparity_map[x, y], 2, -1) * distance
            new_pixel = ((x, y) + res).astype(int)
            if(valid_pixel(new_pixel, (size_x, size_y))):
                novel_view[new_pixel[0], new_pixel[1]] = ref_view[x, y]

def is_same_image(img1, img2):
    """Returns true if img1 has the same values and size as img2, else False"""
    size_x, size_y = img1.shape[0:2]
    size_x1, size_y1 = img2.shape[0:2]
    if(size_x != size_x1 or size_y != size_y1):
        return False

    #Check all pixel values match up
    for x in range(size_x):
        for y in range(size_y):
            arr1 = novel_view[x, y]
            arr2 = novel_view[x, y]
            if(np.all(arr1 != arr2)):
                print('Images different at: {} {}'.format(x, y))
                print('First image value is', arr1)
                print('Second image value is', arr2)
                return False
    return True

def save_array_as_image(array, save_location):
    """Saves an array as an image at the save_location using pillow"""
    image = Image.fromarray(array)
    image.save(save_location)
    image.close()

def main(config):
    hdf5_path = os.path.join(config['PATH']['output_dir'],
                             config['PATH']['hdf5_name'])
    with h5py.File(hdf5_path, mode = 'r', libver = 'latest') as hdf5_file:
        depth_grp = hdf5_file['depth']

        depth_image = depth_grp['images'][0, 27]
        buffer_depth = (depth_image / 255.0).astype(np.float32)
        eye_depth = cs.depth_buffer_to_eye(buffer_depth,
                                           hdf5_file.attrs['near'],
                                           hdf5_file.attrs['far'])
        disparity = cs.depth_to_disparity(eye_depth,
                                          hdf5_file.attrs['baseline'],
                                          hdf5_file.attrs['focal_length'])
        pixel_disp = cs.real_value_to_pixel(disparity,
                                            hdf5_file.attrs['focal_length'],
                                            hdf5_file.attrs['fov'],
                                            depth_grp.attrs['shape'][2])
        #Hardcoded some values for now
        colour_grp = hdf5_file['colour']
        colour_image = colour_grp['images'][0, 27]

        #Can later expand like 0000 if needed
        base_dir = os.path.join(config['PATH']['output_dir'], 'warped')
        get_diff = (config['DEFAULT']['should_get_diff'] == 'True')
        for i in range(8):
            for j in range(8):
                res = fw_warp_image(colour_image, pixel_disp,
                                    np.asarray([3, 3]), np.asarray([i, j]))
                file_name = 'Colour{}{}.png'.format(i, j)
                save_location = os.path.join(base_dir, file_name)
                save_array_as_image(res, save_location)
                if get_diff:
                    colour = colour_grp['images'][0, i * 8 + j]
                    diff = (colour - res).astype(np.uint8)
                    file_name = 'Diff{}{}.png'.format(i, j)
                    save_location = os.path.join(base_dir, file_name)
                    save_array_as_image(diff, save_location)

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read(os.path.join('config','hdf5.ini'))
    dir_to_make = os.path.join(config['PATH']['output_dir'], 'warped')
    if not os.path.exists(dir_to_make):
        os.makedirs(dir_to_make)
    main(config)
