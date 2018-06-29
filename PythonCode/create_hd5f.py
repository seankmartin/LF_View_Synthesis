import configparser
import csv
import os

import h5py
import numpy as np

def main(config):
    hdf5_path = os.path.join(config['PATH']['output_dir'],
                             config['PATH']['hdf5_name'])
    metadata_location = os.path.join(config['PATH']['image_dir'],
                                     config['PATH']['metadata_name'])

    csvfile = open(metadata_location, 'rt')
    reader = csv.reader(csvfile, delimiter = ';', quoting=csv.QUOTE_NONE)
    meta_dict = dict(reader)
    shared_metadata_keys = (
        'baseline', 'near', 'far', 'focal_length', 'fov')

    with h5py.File(hdf5_path, mode = 'w') as hdf5_file:
        for key in shared_metadata_keys:
            hdf5_file.attrs[key] = float(meta_dict[key])

        depth = hdf5_file.create_group('depth')
        # (num_images, grid_size, pixel_width, pixel_height, num_channels)
        depth.attrs['shape'] = [int(config['LF_SIZE']['num_samples']),
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

        csvfile.close()
        hdf5_file.close()
        # Can later be split into train test and val
        #images_shape
        #depth.create_dataset("colour_images", images_shape, np.uint8)
        #colour.create_dataset()

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read(os.path.join('config','hdf5.ini'))
    main(config)
