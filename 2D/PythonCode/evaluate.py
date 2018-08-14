import numpy
import math

from skimage.measure import compare_ssim, compare_psnr

def psnr(img1, img2):
    """Comptutes the PSNR between img1 and img2, max pixel value is 255"""
    return compare_psnr(img1, img2, 255)

def my_psnr(img1, img2):
    """Comptutes the PSNR between img1 and img2, max pixel value is 255"""
    mse = numpy.mean( (img1.astype(float) - img2.astype(float)) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

#See http://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.compare_ssim

def ssim(img1, img2):
    """A wrapper around skimages ssim to match Wang's implementation"""
    return compare_ssim(
        img2.astype(float) / 255.0, img1.astype(float) / 255.0,
        gaussian_weights=True,
        use_sample_covariance=False,
        sigma=1.5,
        multichannel=True,
        data_range=1.0)