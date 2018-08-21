import inviwopy
from inviwopy.glm import vec3

import random
import time

import welford
from lf_camera import LightFieldCamera

def random_f():
    rand_num = random.random()
    if rand_num < 0.5:
        sign = -1
    else: 
        sign = 1
    return sign * random.random()

def random_vec3():
    return vec3(random_f(), random_f(), random_f())

def time_one():
    cam = inviwopy.app.network.EntryExitPoints.camera
    #start_time = time.time()
    lf_camera_here = LightFieldCamera(
        random_vec3(), random_vec3(), random_vec3(),
        interspatial_distance=0.5)

    #Preview the lf camera array
    end_time = lf_camera_here.view_array(cam, save=False, should_time=True)
    #end_time = time.time() - start_time
    print("Rendering complete in {:4f}".format(end_time))
    return end_time

def main(num_samples):
    random.seed(time.time())
    time_accumulator = (0, 0, 0)
    for _ in range(num_samples):
        last_time = time_one()
        time_accumulator = welford.update(time_accumulator, last_time)
    if num_samples > 1:
        mean_time, std_dev_time, _ = welford.finalize(time_accumulator)
        print("Overall time mean: {:4f}, stdev: {:4f}".format(mean_time, std_dev_time))

if __name__ == '__main__':
    NUM_SAMPLES = 5
    main(NUM_SAMPLES)