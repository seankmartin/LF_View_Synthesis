import numpy as np
import matplotlib.pyplot as plt

def main():
    psnr_mean_dict = {
        'no_cnn': 34.094342
        ,'2D': 34.101125
        ,'2D_OneChannel': 33.357591
        ,'3D': 34.453114
        , 'EDSR': 34.478630
        , 'EDSR_remap': 34.338175
    }

    psnr_std_dict = {
        'no_cnn': 2.631714
        ,'2D': 2.643961
        ,'2D_OneChannel': 2.455306
        ,'3D': 2.548295
        , 'EDSR': 2.639852
        , 'EDSR_remap': 2.636725
    }

    ssim_mean_dict = {
        'no_cnn': 0.919227
        ,'2D': 0.919397
        ,'2D_OneChannel': 0.916588
        ,'3D': 0.921789
        , 'EDSR': 0.921591
        , 'EDSR_remap': 0.923498
    }

    ssim_std_dict = {
        'no_cnn': 0.024397
        ,'2D': 0.024459
        ,'2D_OneChannel': 0.024435
        ,'3D': 0.021538
        , 'EDSR': 0.023151
        , 'EDSR_remap': 0.022864
    }

#Image number 43 has the highest pnsr after edsr remap
#Image number 14 has the highest ssim after edsr remap
