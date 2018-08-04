import inviwopy
from inviwopy.data import ImageOutport, ImageInport
from inviwopy.data import Image

import os
import math
import time
import sys

# Location of other scripts
sys.path.insert(0, "/users/pgrad/martins7/.local/lib/python3.5/site-packages")

# Location of torch and torch vision
sys.path.insert(0, "/users/pgrad/martins7/pytorch35/lib/python3.5/site-packages")

import torch
import torchvision.utils as vutils
import numpy as np
from skimage.transform import resize

import conversions
import image_warping

help(inviwopy.properties.BoolProperty)

#help(self)

"""
The PythonScriptProcessor will run this script on construction and whenever this
it changes. Hence one needs to take care not to add ports and properties multiple times.
The PythonScriptProcessor is exposed as the local variable 'self'.
"""

#if not "dim" in self.properties:
#	self.addProperty(IntVec3Property("dim", "dim", ivec3(5), ivec3(0), ivec3(20)))

#if not "outport" in self.outports:
#	self.addOutport(VolumeOutport("outport"))

def process(self):
	"""
    The PythonScriptProcessor will call this process function whenever the processor process 
	function is called. The argument 'self' represents the PythonScriptProcessor.
	"""
	# create a small float volume filled with random noise
	#numpy.random.seed(546465)
	#dim = self.properties.dim.value;
	#volume = Volume(numpy.random.rand(dim[0], dim[1], dim[2]).astype(numpy.float32))
	#volume.dataMap.dataRange = dvec2(0.0, 1.0)
	#volume.dataMap.valueRange = dvec2(0.0, 1.0)
	#self.outports.outport.setData(volume)
	print("Yay!")
	pass

def initializeResources(self):
    pass

# Tell the PythonScriptProcessor about the 'initializeResources' function we want to use
self.setInitializeResources(initializeResources)

# Tell the PythonScriptProcessor about the 'process' function we want to use
self.setProcess(process)