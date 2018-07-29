from random import seed
import os
from time import sleep, time
import pathlib
import math

from lf_camera import LightFieldCamera
from random_lf import create_random_lf_cameras
from random_clip import random_clip_lf, restore_clip
from random_clip import random_plane_clip

import inviwopy
import ivw.utils as inviwo_utils
from inviwopy.glm import vec3, ivec2

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

def save_lf(lf, save_main_dir):
    """Saves a light field in sub dir XXX of save_main_dir"""
    cam = inviwopy.app.network.EntryExitPoints.camera
    sub_dir_to_save_to = get_sub_dir_for_saving(save_main_dir)
    try:
        lf.view_array(cam, save=True,
                      save_dir=sub_dir_to_save_to)
    except ValueError as e:
        print(e)
        os.rmdir(sub_dir_to_save_to)
        lf.view_array(cam)

def main(save_main_dir, pixel_dim, clip, num_random, plane):
    #Setup
    app = inviwopy.app
    network = app.network
    cam = network.EntryExitPoints.camera
    cam.lookUp = vec3(0, 1, 0)
    cam.nearPlane = 6.0
    cam.farPlane = 1000.0
    canvases = inviwopy.app.network.canvases
    for canvas in canvases:
        canvas.inputSize.dimensions.value = ivec2(pixel_dim, pixel_dim)
    inviwo_utils.update()
    if not os.path.isdir(save_main_dir):
        pathlib.Path(save_main_dir).mkdir(parents=True, exist_ok=True)

    #Save a number of random light fields
    random_lfs = create_random_lf_cameras(
                     num_random
                     (180, 35), 1,
                     interspatial_distance=0.5)

    for lf in random_lfs:
        if clip:
            _, clip_type = random_clip_lf(network, lf)
        elif plane:
            random_plane_clip(network, lf)
        save_lf(lf, save_main_dir)
        if clip:
            restore_clip(network, clip_type)

if __name__ == '__main__':
    home = os.path.expanduser('~')
    save_main_dir = os.path.join(home, 'turing', 'overflow-storage', 
                                 'lf_volume_sets', 'test')
    seed(time())
    PIXEL_DIM = 512
    CLIP = False
    PLANE = True
    NUM_RANDOM_LF_SAMPLES = 2
    main(save_main_dir, PIXEL_DIM, CLIP, NUM_RANDOM_LF_SAMPLES, PLANE)
