import numpy as np

def main():
    #Overall time mean: 4.175982, stdev: 0.002914
    #This is in the format mean, stdev, time for only lf grid
    timing_dict = {
        'EDSR_remap': (4.175982, 0.002914, 2.91)
        ,'Direct_render' Overall time mean: 1.134742, stdev: 0.017396
    }

    #For images of size 192 x 192, Srinivasan GPU accelerated warping takes 0.1307 seconds, while CPU warping for with bilinear interp takes 0.17 seconds.