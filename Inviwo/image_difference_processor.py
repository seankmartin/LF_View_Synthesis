"""
Takes in two images from inviwo and outputs the difference between them
"""
import inviwopy
from inviwopy.data import ImageOutport, ImageInport
from inviwopy.data import Image
import numpy as np


INPORT_LIST = ["im_inport1", "im_inport2"]

for name in INPORT_LIST:
    if not name in self.inports:
        self.addInport(ImageInport(name))

if not "outport" in self.outports:
    self.addOutport(ImageOutport("outport"))

def process(self):
    im_data = []
    for name in INPORT_LIST:
        im_data.append(self.getInport(name).getData())
    if not (im_data[0].dimensions == 
            im_data[1].dimensions):
        print("Operation is incompatible with images of different size")
        print("Size 1: ", im_data[0].dimensions)
        print("Size 2: ", im_data[1].dimensions)
        return -1
    out_image = Image(im_data[0].dimensions, 
                      inviwopy.data.formats.DataVec4UINT8)
    print(out_image.colorLayers[0].data.shape)
    im_colour = []
    for idx, name in enumerate(INPORT_LIST):
        im_colour.append(im_data[idx].colorLayers[0].data)
    
    out_colour = get_diff_image(im_colour[0], im_colour[1])
    print(out_colour.shape)
    out_image.colorLayers[0].data = out_colour
   
    im_depth = []
    for idx, name in enumerate(INPORT_LIST):
        im_depth.append(im_data[idx].depth.data)

    out_depth = get_diff_image(im_depth[0], im_depth[1], np.float32)
    out_image.depth.data = out_depth

    self.getOutport("outport").setData(out_image)

def initializeResources(self):
	pass

def get_diff_image(im1, im2, dtype=np.uint8):
    diff = np.subtract(im1.astype(float), im2.astype(float))
    diff = abs(diff).astype(dtype)
    return diff

# Tell the PythonScriptProcessor about the 'initializeResources' function we want to use
self.setInitializeResources(initializeResources)

# Tell the PythonScriptProcessor about the 'process' function we want to use
self.setProcess(process)