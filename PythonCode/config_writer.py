#A script for easily changing the config file
import configparser
import os

#Change configurations here
config = configparser.ConfigParser()
config['LF_SIZE'] = {'GridSize' : '8',
                     'ImageWidth' : '256',
                     'ImageHeight' : '256'}

#Write to the file
filename = 'sample.ini'
here = os.path.dirname(os.path.abspath(__file__))
dir_to_make = os.path.join(here, 'config')

if not os.path.exists(dir_to_make):
    os.makedirs(dir_to_make)

relative_filepath = os.path.join(here, 'config', filename)
with open(relative_filepath, 'w') as configfile:
    config.write(configfile)
