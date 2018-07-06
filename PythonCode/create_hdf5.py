import configparser
import csv
import os

import h5py
import numpy as np
from PIL import Image

import welford

def get_row_col_number(index, spatial_cols):
    """Turns index into (x,y) grid reference with spatial_cols grid columns"""
    row_num = index // spatial_cols
    col_num = index % spatial_cols
    return row_num, col_num

def index_num_to_grid_loc(index, spatial_cols):
    """Turns index into 'xy' grid reference with spatial_cols grid columns"""
    row_num, col_num = get_row_col_number(index, spatial_cols)
    image_num_str = str(row_num) + str(col_num)
    return image_num_str

def main(config):
    hdf5_path = os.path.join(config['PATH']['output_dir'],
                             config['PATH']['hdf5_name'])
    base_dir = config['PATH']['image_dir']
    metadata_location = os.path.join(base_dir,
                                     config['PATH']['metadata_name'])
    sub_dirs = [os.path.join(base_dir, el)
                    for el in os.listdir(base_dir)
                    if os.path.isdir(os.path.join(base_dir, el))]

    csvfile = open(metadata_location, 'rt')
    reader = csv.reader(csvfile, delimiter = ';', quoting=csv.QUOTE_NONE)
    meta_dict = dict(reader)
    shared_metadata_keys = (
        'baseline', 'near', 'far', 'focal_length', 'fov')

    with h5py.File(hdf5_path, mode = 'w', libver = 'latest') as hdf5_file:
        for key in shared_metadata_keys:
            hdf5_file.attrs[key] = np.float32(meta_dict[key])

        depth = hdf5_file.create_group('depth')
        # (num_images, grid_size, pixel_width, pixel_height, num_channels)
        depth.attrs['shape'] = [len(sub_dirs),
                                int(meta_dict['grid_rows']) *
                                int(meta_dict['grid_cols']),
                                int(meta_dict['pixels']),
                                int(meta_dict['pixels']),
                                1]

        colour = hdf5_file.create_group('colour')
        # Same shape as depth but with more channels
        temp = depth.attrs['shape']
        temp[4] = 3
        colour.attrs['shape'] = temp
        depth_image_shape = [len(sub_dirs),
            	             int(meta_dict['pixels']),
                             int(meta_dict['pixels']),
                             1]
        temp = np.copy(depth_image_shape)
        temp[3] = 3
        colour_image_shape = temp

        #Save the images:
        #Can later be split into train test and val
        dim = int(meta_dict['pixels'])
        depth.create_dataset('images', depth.attrs['shape'], np.uint8,
                             chunks = (1, 1, dim, dim, 1),
                             compression = "lzf",
                             shuffle = True)
        depth.create_dataset('mean', depth_image_shape)
        depth.create_dataset('var', depth_image_shape)
        colour.create_dataset('images', colour.attrs['shape'], np.float32,
                              chunks = (1, 1, dim, dim, 3),
                              compression = "lzf",
                              shuffle = True)
        colour.create_dataset('mean',colour_image_shape)
        colour.create_dataset('var', colour_image_shape)

        cols = int(meta_dict['grid_cols'])
        size = int(meta_dict['grid_rows']) * int(meta_dict['grid_cols'])
        for idx, dir in enumerate(sub_dirs):
            print(dir)
            depth_mean = np.zeros(depth_image_shape[1:-1], np.float32)
            colour_mean = np.zeros(colour_image_shape[1:], np.float32)
            depth_accumulator = (0, depth_mean, 0)
            colour_accumulator = (0, colour_mean, 0)

            for x in range(size):
                image_num = index_num_to_grid_loc(x, cols)
                depth_name = 'Depth' + image_num + '.png'
                depth_loc = os.path.join(dir, depth_name)
                depth_image = Image.open(depth_loc)
                depth_image.load()
                depth_data = np.asarray(depth_image, dtype = np.uint8)
                depth['images'][idx, x, :, :, 0] = depth_data
                depth_accumulator = (
                    welford.update(depth_accumulator, depth_data))

                colour_name = 'Colour' + image_num + '.png'
                colour_loc = os.path.join(dir, colour_name)
                colour_image = Image.open(colour_loc)
                colour_image.load()
                colour_data = np.asarray(colour_image, dtype = np.uint8)
                colour['images'][idx, x, :, :, :] = colour_data[:, :, :3]
                colour_accumulator = (
                    welford.update(colour_accumulator, colour_data[:, :, :3]))

            (depth['mean'][idx, :, :, 0],
             depth['var'][idx, :, :, 0], _) = (
                welford.finalize(depth_accumulator))
            (colour['mean'][idx, :, :, :],
             colour['var'][idx, :, :, :], _) = (
                welford.finalize(colour_accumulator))

        csvfile.close()
        hdf5_file.close()
    print("Finished writing to", hdf5_path)

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read(os.path.join('config','hdf5.ini'))
    main(config)
