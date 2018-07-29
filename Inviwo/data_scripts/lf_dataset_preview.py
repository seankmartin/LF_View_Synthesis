from random import seed, random
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

def main(pixel_dim, clip, num_random, plane):
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

    random_lfs = create_random_lf_cameras(
                     num_random,
                    (180, 35), 1,
                     interspatial_distance=0.5)
    for lf in random_lfs:
        if clip:
            _, clip_type = random_clip_lf(network, lf)
        elif plane:
            random_plane_clip(network, lf)
        lf.view_array(cam, save=False)
        if clip:
            restore_clip(network, clip_type)

if __name__ == '__main__':
    seed(time())
    PIXEL_DIM = 512
    CLIP = False
    PLANE = True
    NUM_RANDOM_LF_SAMPLES = 2
    
    main(PIXEL_DIM, CLIP, NUM_RANDOM_LF_SAMPLES, PLANE)
