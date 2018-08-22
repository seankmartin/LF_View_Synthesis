"""
Takes in two images from inviwo and outputs the difference between them
"""
import os
import math
import time
import sys
import warnings

# Location of torch and torch vision
sys.path.insert(0, "/users/pgrad/martins7/pytorch35/lib/python3.5/site-packages")

# Location of other scripts
#sys.path.insert(0, "/users/pgrad/martins7/.local/lib/python3.5/site-packages")

# Location of pythonCode
sys.path.insert(0, "/users/pgrad/martins7/LF_View_Synthesis/Direct_3D/PythonCode")

import inviwopy
from inviwopy.data import ImageOutport, ImageInport
from inviwopy.properties import BoolProperty
from inviwopy.properties import IntProperty
from inviwopy.data import Image

import torch
import torchvision.utils as vutils
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte

import conversions
import image_warping
import data_transform
import cnn_utils

# For each input image to the network, add an inport here
INPORT_LIST = ["im_inport1"]
model = None
cuda = True
GRID_SIZE = 64
SIZE = 1024
OUT_SIZE = inviwopy.glm.size2_t(SIZE, SIZE)
SAMPLE_SIZE = inviwopy.glm.size2_t(512, 512)
SAMPLE_SIZE_LIST = [512, 512]
OUT_SIZE_LIST = [SIZE, SIZE]
DTYPE = inviwopy.data.formats.DataVec4UINT8

if model is None:
    """Load the model to be used"""
    model_dir = "/users/pgrad/martins7/turing/overflow-storage/outputs/models"
    name = "best_direct_model.pth"

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

if not "sample" in self.outports:
    self.addOutport(ImageOutport("sample"))

if not "off" in self.properties:
    self.addProperty(
        BoolProperty("off", "Off", True))

if not "display_input" in self.properties:
    self.addProperty(
        BoolProperty("display_input", "Show Input", True))

if not "sample_num" in self.properties:
    self.addProperty(
        IntProperty("sample_num", "Sample Number", 0)
    )
    self.getPropertyByIdentifier("sample_num").minValue = 0
    self.getPropertyByIdentifier("sample_num").maxValue = 63

def process(self):
    """Perform the model warping and output an image grid"""
    if self.getPropertyByIdentifier("off").value:
        print("Image warping is currently turned off")
        return 1

    start_time = time.time()

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
    sample_image = Image(SAMPLE_SIZE, DTYPE)
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
        'depth': torch.tensor(im_depth[0], dtype=torch.float32).unsqueeze_(0), 
        'colour': torch.tensor(im_colour[0], dtype=torch.float32).unsqueeze_(0),
        'grid_size': GRID_SIZE}

    warped = data_transform.inviwo_central(sample)
    im_input = warped['inputs'].unsqueeze_(0)
    
    if cuda:
        im_input = im_input.cuda()

    model.eval()
    output = model(im_input)
    
    end_time = time.time() - start_time
    print("Grid light field rendered in {:4f}".format(end_time))
    out_final = output[0]
    out_colour = cnn_utils.transform_lf_to_torch(out_final)

    output_grid = vutils.make_grid(
                    out_colour, nrow=8, range=(-1, 1), normalize=True,
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
    
    shape = tuple(SAMPLE_SIZE_LIST) + (4,)
    sample_out = np.full(shape, 255, np.uint8)
    sample_out[:, :, :3] = np.around(
        data_transform.denormalise_lf(
            out_final).cpu().detach().numpy()
    ).astype(np.uint8)[self.getPropertyByIdentifier("sample_num").value]

    # Inviwo expects a uint8 here
    out_image.colorLayers[0].data = final_out
    sample_image.colorLayers[0].data = sample_out
    self.getOutport("outport").setData(out_image)
    self.getOutport("sample").setData(sample_image)

    end_time = time.time() - start_time
    print("Overall render time was {:4f}".format(end_time))

def initializeResources(self):
    pass

# Tell the PythonScriptProcessor about the 'initializeResources' function we want to use
self.setInitializeResources(initializeResources)

# Tell the PythonScriptProcessor about the 'process' function we want to use
self.setProcess(process)
