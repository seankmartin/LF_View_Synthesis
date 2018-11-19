# LF_View_Synthesis

M.Sc. Dissertation project to synthesise light fields from sample renderings of volumetric data using deep learning.
Depth maps are produced during volume rendering at the ray tracing step using heuristics.
Using these maps images are warped from reference views to novel positions.
A CNN is used to account for inevitable errors in the depth map and non-Lambertian effects.

This repository is structured so that the folders 2D, 2d_OneChannel, 3D, DepthEstimate_angular2D, Direct_3D, Pretrained_2D and Pretrained_angular2D all represent different models tested.
From the paper 2D is StackedResNet, Pretrained_2D is StackedEDSR, Pretrained_angular2d is AngularEDSR.
Inside each of these folders, PythonCode contains the python files, Inviwo contains the files needed to run that network in Inviwo, and Scripts contains bash scripts for convienience.

## Replicating data capture

[Inviwo](http://www.inviwo.org/) is required to produce the test data. On a Windows or Mac system binaries can be downloaded from http://www.inviwo.org/download/application/. Linux users are required to build Inviwo from source with cmake, see [The Inviwo Github wiki](https://github.com/inviwo/inviwo/wiki/Building-Inviwo-on-Linux). Note that on Linux, you may run into an error with cmake. This could be because the cmake cache file needs to be updated to point to the cmake lib file in the qt install directory, which happened to me. The Inviwo wiki has some information on this at [Cmake tips](https://github.com/inviwo/inviwo/wiki/CMake-Tips).

### To build Inviwo with the custom modules I created installed, follow these steps:

1. Clone my fork of Inviwo `git clone --recurse-submodules https://github.com/flabby99/inviwo`
2. Generate files and build Inviwo:
    - For convience, a build script for linux is provided in tools/inviwo_build.sh in my fork   of Inviwo.
    - Place it in the parent directory of the inviwo source code cloned from github.
    - Note that the number of cores used for building must be changed in the script, by         default it is 8.
    - Make sure to pass the location of the Qt cmake directory to the script, see [Inviwo       Wiki](https://github.com/inviwo/inviwo/wiki/Building-Inviwo-on-Linux) on how to install   Qt. For example, home/Qt5.6.1/5.6/gcc_64/lib/cmake is the location of this in my case.
    - If building on Windows, the process is very similar, and the script provided for linux    should help. However, cmake should produce visual studio files.
3. Verify modules correctly installed by opening inviwo/data/workspaces/myworkspaces/depth_heuristics.inv in your built version of Inviwo.

### Once Inviwo is installed, open up a terminal or command prompt and follow these steps

1. Move to the bin directory in the inviwo install directory in a terminal or command prompt.
2. Open Inviwo with the command: `./inviwo -w ~/inviwo/data/workspaces/data_capture.inv` (assuming inviwo was cloned to the home directory)
5. Ensure that the volume source processor is reading from a valid location. If not, change it to one of the volumes provided by Inviwo.
6. The data capture can be previewed with ``./inviwo -w ~/inviwo/data/workspaces/data_capture.inv -p ~/inviwo/pyscripts/data_scripts/lf_dataset_preview.py``
7. Or captured with ``./inviwo -w ~/inviwo/data/workspaces/data_capture.inv -p ~/inviwo/pyscripts/data_scripts/lf_dataset_capture.py``

### Processsing to a hdf5 file

Update hdf5.ini in 2D/PythonCode/config to contain valid input and output locations for the data. Running python3 create_hdf5 will process all the captured data in Inviwo into a hdf5 file for use in training and validating the CNN. This can also be used for image warping.

## Replicating image warping

The data capture script from Inviwo saves depth maps for each light field view location. Using the central depth map, the pixels of any view can be warped to any other view by converting the depth to disparity.

PythonCode folders have:

``conversions.py`` has a function to convert a depth map (represented by a numpy array) to a disparity map in pixels.

``image_warping.py`` contains functions to perform image warping with the above disparity map.

These function the same for all network architectures.

## Replicating Convolutional Neural Network

[Pytorch](https://pytorch.org/) is required to run the CNN. Furthermore, a requirements.txt file is provided for convenience. This can be installed in a python virtual environment by running `pip3 install -r requirements.txt.`

### Validation

Run the cnn_demo.py file inside of any PythonCode folder to perform validation. This requires a hdf5 file and a trained model to function.

### Training

Run the cnn_main file in PythonCode, or if using Linux, moving to the Scripts directory and running ``./run_models.sh "arguments for training run 1" "arguments for training run 2" ... "arguments required for training run n"`` is probably more convienient. In either case, the -h flag gives information on command line arguments that can be passed, and main.ini provides a config for path locations and network parameters.
