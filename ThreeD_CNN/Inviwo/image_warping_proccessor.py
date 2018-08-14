"""
Takes in two images from inviwo and outputs the difference between them
"""
import os
import math
import time
import sys
import warnings

# Location of other scripts
sys.path.insert(0, "/users/pgrad/martins7/.local/lib/python3.5/site-packages")

# Location of torch and torch vision
sys.path.insert(0, "/users/pgrad/martins7/pytorch35/lib/python3.5/site-packages")

# Location of pythonCode
sys.path.insert(0, "/users/pgrad/martins7/LF_View_Synthesis/ThreeD_CNN/PythonCode")

import inviwopy
from inviwopy.data import ImageOutport, ImageInport
from inviwopy.properties import BoolProperty
from inviwopy.data import Image

import torch
import torchvision.utils as vutils
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte

import conversions
import image_warping

# For each input image to the network, add an inport here
INPORT_LIST = ["im_inport1"]
model = None
cuda = True
GRID_SIZE = 64
SIZE = 1024
OUT_SIZE = inviwopy.glm.size2_t(SIZE, SIZE)
OUT_SIZE_LIST = [SIZE, SIZE]
DTYPE = inviwopy.data.formats.DataVec4UINT8

if model is None:
    """Load the model to be used"""
    model_dir = "/users/pgrad/martins7/overflow-storage/outputs/models"
    name = "best_model.pth"

    #Load model architecture
    model = torch.load(os.path.join(
            model_dir, name))['model']

    # Load the weipreserve_range=Falseghts into the model
    weights_location = os.path.join(
            model_dir, name)
    if os.path.isfile(weights_location):
        print("=> loading model '{}'".format(weights_location))
        weights = torch.load(weights_location)
        model.load_state_dict(weights['model'].state_dict())
        print("=> model successfully loaded")
    else:
        print("=> no model found at '{}'".format(weights_location))

    if cuda:
        model = model.cuda()

    model.eval()

for name in INPORT_LIST:
    if not name in self.inports:
        self.addInport(ImageInport(name))

if not "outport" in self.outports:
    self.addOutport(ImageOutport("outport"))

if not "off" in self.properties:
    self.addProperty(
        BoolProperty("off", "Off", True))

if not "display_input" in self.properties:
    self.addProperty(
        BoolProperty("display_input", "Show Input", True))

def process(self):
    """Perform the model warping and output an image grid"""
    if self.getPropertyByIdentifier("off").value:
        print("Image warping is currently turned off")
        return 1

    if self.getPropertyByIdentifier("display_input").value:
        im_data = []
        for name in INPORT_LIST:
            im_data.append(self.getInport(name).getData())
        out_image = Image(OUT_SIZE, DTYPE)
        out = resize(
            im_data[0].colorLayers[0].data.transpose(1, 0, 2),
            OUT_SIZE_LIST)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            inter_out = img_as_ubyte(out)
        out_image.colorLayers[0].data = inter_out
        self.getOutport("outport").setData(out_image)
        return 1

    if model is None:
        print("No model for synthesis")
        return -1

    cam = inviwopy.app.network.EntryExitPoints.camera
    im_data = []
    for name in INPORT_LIST:
        im_data.append(self.getInport(name).getData())
    for im in im_data: 
        if not (im_data[0].dimensions == im.dimensions):
            print("Operation is incompatible with images of different size")
            print("Size 1: ", im_data[0].dimensions)
            print("Size 2: ", im.dimensions)
            return -1
     
    out_image = Image(OUT_SIZE, DTYPE)
    im_colour = []
    for idx, name in enumerate(INPORT_LIST):
        im_colour.append(im_data[idx].colorLayers[0].data[:, :, :3].transpose(1, 0, 2))
    
    im_depth = []
    near = cam.nearPlane
    far = cam.farPlane
    baseline = 0.5
    focal_length = cam.projectionMatrix[0][0]
    fov = cam.fov.value
    
    for idx, name in enumerate(INPORT_LIST):
        im_depth.append(
            conversions.depth_to_pixel_disp(
                im_data[idx].depth.data.transpose(1, 0),
                near=near, far=far, baseline=baseline,
                focal_length=focal_length,
                fov=fov,
                image_pixel_size=float(im_data[0].dimensions[0]))
        )
    
    sample = {
        'depth': torch.tensor(im_depth[0], dtype=torch.float32), 
        'colour': torch.tensor(im_colour[0], dtype=torch.float32), 
        'grid_size': GRID_SIZE}

    warped = transform_to_warped(sample)
    im_input = warped['inputs'].unsqueeze_(0)
    im_input.requires_grad_(False)
    
    if cuda:
        im_input = im_input.cuda()

    model.eval()
    output = model(im_input)
    output += im_input
    output = torch.clamp(output, 0.0, 1.0)

    out_colour = transform_lf_to_torch(output[0])

    output_grid = vutils.make_grid(
                    out_colour, nrow=8, range=(0, 1), normalize=False,
                    padding=2, pad_value=1.0)

    output_grid = resize(
        output_grid.cpu().detach().numpy().transpose(1, 2, 0),
        OUT_SIZE_LIST)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inter_out = img_as_ubyte(output_grid)

    #inter_out = denormalise_lf(output_grid)
    #inter_out = inter_out.cpu().detach().numpy().astype(np.uint8).transpose(1, 2, 0)
    # Add an alpha channel here
    shape = tuple(OUT_SIZE_LIST) + (4,)
    final_out = np.full(shape, 255, np.uint8)
    final_out[:, :, :3] = inter_out
    
    # Inviwo expects a uint8 here
    out_image.colorLayers[0].data = final_out
    self.getOutport("outport").setData(out_image)

def initializeResources(self):
    pass

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
    for i in range(grid_one_way):
        for j in range(grid_one_way):
            res = image_warping.fw_warp_image(
                ref_view=views,
                disparity_map=disparities,
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
