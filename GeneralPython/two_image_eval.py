import argparse

import numpy as np
from PIL import Image

import evaluate


def main(args):
    if args.im1 is None or args.im2 is None:
        print("Please enter two image paths through cmd")
        exit(-1)
    print("Comparing {} and {}".format(args.im1, args.im2))
    im1 = Image.open(args.im1)
    im2 = Image.open(args.im2)
    im1 = np.array(im1)
    im2 = np.array(im2)
    psnr = evaluate.my_psnr(im1, im2)
    ssim = evaluate.ssim(im1, im2)
    print("PSNR {:4f}, SSIM {:4f}".format(psnr, ssim))

if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(description="modifiable params")
    PARSER.add_argument("--im1", type=str, default=None,
                        help="first image path for comparison")
    PARSER.add_argument("--im2", type=str, default=None,
                        help="second image path for comparison")   
    ARGS, _ = PARSER.parse_known_args()
    main(ARGS)