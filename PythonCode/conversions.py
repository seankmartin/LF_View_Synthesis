"""Coverts depth buffer information to pixel disparity"""
from math import radians, tan

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

def depth_to_disparity(depth_value, baseline, focal_length, shift=0.0):
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

def real_value_to_pixel(real_value, focal_length, fov, image_pixel_size):
    """
    Converts a value in real units to a pixel value by similar triangles

    Keyword arguments:
    focal_length -- camera focal length in real units
    fov -- camera field of view in degrees
    image_pixel_size -- the width of the image in pixels
    """

    image_sensor_size = 2 * focal_length * tan(radians(fov / 2.0))
    return real_value * image_pixel_size / image_sensor_size

def depth_to_pixel_disp(depth, near, far, baseline, focal_length, fov, 
                        image_pixel_size, shift=0.0):
    """
    Performs the whole pipeline for converting buffer depth to pixel disparity
    """
    inviwo_depth = depth_buffer_to_eye(depth, near, far)
    disparity = depth_to_disparity(inviwo_depth, baseline, focal_length)
    pixel_disparity = real_value_to_pixel(disparity, focal_length, fov, pixels)
    return pixel_disparity

def example(input_depth):
    """Returns information on calculations with fixed intrinsics"""
    input_depth = input_depth
    baseline = 0.5
    near = 6
    far = 686
    focal_length = 1.545
    fov = 66
    pixels = 512
    inviwo_depth = depth_buffer_to_eye(input_depth, near, far)
    disparity = depth_to_disparity(inviwo_depth, baseline, focal_length)
    final = real_value_to_pixel(disparity, focal_length, fov, pixels)
    print("Buffer depth is {:4f}".format(input_depth))
    print("Inviwo depth is {:4f}".format(inviwo_depth))
    print("Disparity is {:4f}".format(disparity))
    print("Pixel disparity is {:4f}".format(final))


if __name__ == '__main__':
    example(0.901) # expect roughly 3.5 as output
    while True:
        try:
            print("Please enter q to quit, or another value to try again")
            STR = input('--> ')
            if STR == "q":
                break
            else:
                example(float(STR))
        except ValueError:
            print("No input read, assuming you wanted to exit")
            exit(0)
