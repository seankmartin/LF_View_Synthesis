import configparser

from PIL import Image
import numpy as np

def depth_buffer_to_eye(buffer_depth, near, far):
    """
    Returns eye space depth from a [0,1] depth buffer value.
    First converts the buffer value to NDC in [-1, 1].
    Then inverts the perspective projection to give eye space depth.

    Keyword arguments:
    buffer_depth -- the input depth buffer value
    near -- the depth of the near plane (positive)
    far -- the depth of the far plane (positive)
    """
    ndc_depth = 2.0 * buffer_depth - 1.0
    eye_depth = 2.0 * near * far / (near + far - ndc_depth * (far - near))
    return eye_depth

def depth_to_disparity(depth_value, baseline, focal_length, shift = 0.0):
    """
    Returns a disparity value from a depth value

    Keyword arguments:
    depth_value -- the input depth
    baseline -- the distance between neighbouring cameras in the grid
    focal_length -- the depth image capturing camera's focal length
    shift -- an optional distance between neighbouring images
          -- principal point offsets (default 0.0),
          -- note that this is from the HCI paper on their capture process
    """
    disparity = (baseline * focal_length) / depth_value - shift
    return disparity

def fw_warp_image(ref_view, disparity_map, ref_pos, novel_pos):
    """
    Returns a forward warped novel from an input image and disparity_map
    For each pixel position in the reference view, shift it by the disparity,
    and assign the value in the reference at that new pixel position to the
    novel view.

    Keyword arguments:
    ref_view -- colour image data at the reference position
    disparity_map -- a disparity map at the reference position
    ref_pos -- the grid co-ordinates of the ref_view
    novel_pos -- the target grid position for the novel view
    """
    size_x, size_y = ref_view.shape[0:2]

    #Create an array of pixel positions
    grid = np.meshgrid(np.arange(size_x),np.arange(size_y))
    stacked = np.stack(grid, 2)
    pixels = stacked.reshape(-1, 2)

    distance = ref_pos - novel_pos
    #Repeat the elements of the disparity_map to match the distance
    repeated = np.repeat(disparity_map, 2, -1).reshape((size_x, size_y, 2))
    result = repeated * distance

    novel_pixels = pixels + result

    for

def main(config):



if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read(os.path.join('config','hdf5.ini'))
    main(config)
