from random import random

from inviwopy.glm import vec3, normalize

from lf_camera import LightFieldCamera

def random_float():
    """returns a random float between -1 and 1"""
    return (random() - 0.5) * 2

def rand_vec_in_unit_sphere():
    return vec3(random_float(), random_float(), random_float())

def rand_vec_between_spheres(big_radius, small_radius):
    """
    Generates a vec3 in between the shell of two spheres
    Input is the radius of the big and small sphere
    """
    radius_diff = big_radius - small_radius
    point_on_unit_sphere = normalize(rand_vec_in_unit_sphere())
    scale_factor = (random() * radius_diff + small_radius)
    point_in_between = point_on_unit_sphere * scale_factor    
    return point_in_between

def create_random_lf_cameras(num_to_create, look_from_radii,
                             max_look_to_origin=1.0,
                             interspatial_distance=1.0,
                             spatial_rows=8, spatial_cols=8):
    """Create a list of randomnly positioned lf cameras

    Keyword arguments:
    num_to_create -- the number of lf cameras to create
    look_from_radii -- min, max distance from the origin to camera look from
    max_look_to_origin -- max distance from the origin to camera look to    
    interspatial_distance -- distance between cameras in array (default 1.0)
    spatial_rows, spatial_cols -- dimensions of camera array (default 8 x 8)
    """
    lf_cameras = []
    d = max_look_to_origin
    look_up = vec3(0, 1, 0)
    for _ in range(num_to_create):
        look_from = rand_vec_between_spheres(*look_from_radii)
        look_to = vec3(random_float(), random_float(), random_float()) * d
        lf_cam = LightFieldCamera(look_from, look_to, look_up,
                                  interspatial_distance,
                                  spatial_rows, spatial_cols)
        lf_cameras.append(lf_cam)
    return lf_cameras