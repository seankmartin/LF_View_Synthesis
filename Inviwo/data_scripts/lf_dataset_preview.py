from random import seed, random
import os
from time import sleep, time
import pathlib
import math

from lf_camera import LightFieldCamera
from random_lf import create_random_lf_cameras
from random_clip import random_subset, restore_volume

import inviwopy
import ivw.utils as inviwo_utils
from inviwopy.glm import vec3, ivec2

def main(save_main_dir, pixel_dim):
    #Setup
    app = inviwopy.app
    network = app.network
    cam = network.EntryExitPoints.camera
    #cam.lookTo = vec3(0, 0, 0)
    cam.lookUp = vec3(0, 1, 0)
    cam.nearPlane = 6.0
    cam.farPlane = 1000.0
    canvases = inviwopy.app.network.canvases
    for canvas in canvases:
        canvas.inputSize.dimensions.value = ivec2(pixel_dim, pixel_dim)
    inviwo_utils.update()

    # Create a light field camera at the current camera position
    # lf_camera_here = LightFieldCamera(cam.lookFrom, cam.lookTo,
    #                                   interspatial_distance=0.5)

    #Preview the lf camera array
    #lf_camera_here.view_array(cam, save=False)

    #Save a number of random light fields
    NUM_RANDOM_LF_SAMPLES = 2
    CLIP_TYPE = "Z"
    random_lfs = create_random_lf_cameras(
                     NUM_RANDOM_LF_SAMPLES, 
                    (200, 35), 1,
                     interspatial_distance=0.5)
    random_subset(network, CLIP_TYPE)
    for lf in random_lfs:
        lf.view_array(cam, save=False)
    restore_volume(network, CLIP_TYPE)

if __name__ == '__main__':
    home = os.path.expanduser('~')
    #home = 'E:'
    save_main_dir = os.path.join(home, 'turing', 'overflow-storage', 
                                 'lf_volume_sets', 'test')
    seed(time())
    PIXEL_DIM = 512
    main(save_main_dir, PIXEL_DIM)
