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

    im_colour = []
    for idx, name in enumerate(INPORT_LIST):
        im_colour.append(im_data[idx].colorLayers[0].data)
    
    out_colour = get_diff_image(im_colour[0], im_colour[1])
    out_image = Image.clone(im_data[0])
    out_image.colorLayers[0].data = out_colour
    self.getOutport("outport").setData(out_image)

def initializeResources(self):
	pass

def get_diff_image(im1, im2):
    diff = np.subtract(im1.astype(float), im2.astype(float))
    diff = abs(diff).astype(np.uint8)
    return diff

# Tell the PythonScriptProcessor about the 'initializeResources' function we want to use
self.setInitializeResources(initializeResources)

# Tell the PythonScriptProcessor about the 'process' function we want to use
self.setProcess(process)