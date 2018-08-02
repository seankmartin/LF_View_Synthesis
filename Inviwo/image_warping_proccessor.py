"""
Takes in two images from inviwo and outputs the difference between them
"""
import os
import math
import time

import inviwopy
from inviwopy.data import ImageOutport, ImageInport
from inviwopy.data import Image

import torch
import torchvision.utils as vutils
import numpy as np
from skimage.transform import resize

import image_warping

# For each input image to the network, add an inport here
INPORT_LIST = ["im_inport1"]
model = None
cuda = True
GRID_SIZE = 64
OUT_SIZE = inviwopy.glm.size2_t(1024, 1024)
DTYPE = inviwopy.data.formats.DataUINT8

for name in INPORT_LIST:
    if not name in self.inports:
        self.addInport(ImageInport(name))

if not "outport" in self.outports:
    self.addOutport(ImageOutport("outport"))

def process(self):
    """Perform the model warping and output an image grid"""

    im_data = []
    for name in INPORT_LIST:
        im_data.append(self.getInport(name).getData())
    for im in im_data: 
        if (im_data[0].dimensions is not im.dimensions):
            print("Operation is incompatible with images of different size")
            print("Size 1: ", im_data[0].dimensions)
            print("Size 2: ", im.dimensions)
            return -1
     
    out_image = Image(OUT_SIZE, DTYPE)
    im_colour = []
    for idx, name in enumerate(INPORT_LIST):
        im_colour.append(im_data[idx].colorLayers[0].data)
    
    im_depth = []
    for idx, name in enumerate(INPORT_LIST):
        im_depth.append(im_data[idx].depth.data)
    
    # TODO convert depth to disparity
    sample = {'depth': im_depth, 
              'colour': im_colour,
              'grid_size': GRID_SIZE}

    warped = transform_to_warped(sample)
    im_input = warped['inputs'].unsqueeze_(0)

    if cuda:
        im_input = im_input.cuda()

    output = model(im_input)
    output += im_input
    output = torch.clamp(output, 0.0, 1.0)

    out_colour = transform_lf_to_torch(output)
    # Inviwo expects a uint8 here
    output_grid = vutils.make_grid(
                    out_colour, nrow=8, range=(0, 1), normalize=True,
                    pad_value=1.0)
    output_grid = resize(
        output_grid.transpose(1, 2, 0),
        OUT_SIZE,
        preserve_range=True)

    # Add an alpha channel here
    shape = OUT_SIZE + (4,)
    final_out = np.ones(shape, np.uint8)
    final_out[:, :, :3] = output_grid
    out_image.colorLayers[0].data = final_out
    
    self.getOutport("outport").setData(out_image)

def initializeResources(self):
    """Load the model to be used"""
    model_dir = "/users/pgrad/martins7/overflow-storage/outputs/models"
    name = "best_model.pth"

    #Load model architecture
    model = torch.load(os.path.join(
            model_dir, name))['model']

    # Load the weights into the model
    weights_location = os.path.join(
            model_dir, name)
    if os.path.isfile(weights_location):
        print("=> loading model '{}'".format(weights_location))
        weights = torch.load(weights_location)
        model.load_state_dict(weights['model'].state_dict())
    else:
        print("=> no model found at '{}'".format(weights_location))

    if cuda:
        model = model.cuda()

    model.eval()

def transform_lf_to_torch(lf):
    """
    Torch expects a grid of images to be in the format:
    (B x C x H x W) but our light field grids are in the format:
    (B x W x H x C) so need to transpose them
    """
    return lf.transpose(1, 3).transpose(2, 3)

def denormalise_lf(lf):
    """Coverts an lf in the range 0 1 to 0 to maximum"""
    maximum = 255.0
    lf.mul_(maximum)
    return lf

def disparity_based_rendering(
        disparities, views, grid_size,
        dtype=np.float32, blank=-1.0):
    """Returns a list of warped images using the input views and disparites"""
     # Alternatively, grid_one_way - 1 can be used below
    shape = (grid_size,) + views.shape[-3:]
    warped_images = np.empty(
        shape=shape, dtype=dtype)
    grid_one_way = int(math.sqrt(grid_size))
    sample_index = 0
    for i in range(grid_one_way):
        for j in range(grid_one_way):
            res = image_warping.fw_warp_image(
                ref_view=views[sample_index],
                disparity_map=disparities[sample_index],
                ref_pos=np.asarray([grid_one_way // 2, grid_one_way // 2]),
                novel_pos=np.asarray([i, j]),
                dtype=dtype,
                blank=blank
            )
            warped_images[i * grid_one_way + j] = res
    return warped_images

def transform_to_warped(sample):
    """
    Input a dictionary of depth images and reference views,
    Output a dictionary of inputs -warped and targets - reference
    """
    normalise_sample(sample)
    disparity = sample['depth']
    targets = sample['colour']
    grid_size = sample['grid_size']
    warped_images = disparity_based_rendering(
        disparity.numpy(), targets.numpy(), grid_size,
        dtype=np.float32, blank=0.0)

    inputs = torch.from_numpy(warped_images)
    return {'inputs': inputs, 'targets': targets}

def normalise_sample(sample):
    """Coverts an lf in the range 0 to maximum into 0 1"""
    maximum = 255.0
    lf = sample['colour']
    lf.div_(maximum)
    return sample

# Tell the PythonScriptProcessor about the 'initializeResources' function we want to use
self.setInitializeResources(initializeResources)

# Tell the PythonScriptProcessor about the 'process' function we want to use
self.setProcess(process)