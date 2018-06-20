# LF_View_Synthesis
M.Sc. Dissertation project to synthesise light fields from sample renderings of volumetric data using deep learning

# Replicating data capture
[Inviwo](http://www.inviwo.org/) is required to produce the test data. On a Windows or Mac system binaries can be downloaded from http://www.inviwo.org/download/application/. Linux users are required to build Inviwo from source with cmake, see [The Inviwo Github wiki](https://github.com/inviwo/inviwo/wiki/Building-Inviwo-on-Linux). Note that on Linux, you may run into an error with cmake. This could be because the cmake cache file needs to be updated to point to the cmake lib file in the qt install directory, which happened to me.

Once Inviwo is installed, open up a terminal or command prompt and follow these steps: 
1. Move to the bin directory in the inviwo install directory in a terminal or command prompt. 
2. Open Inviwo with the command: ./inviwo -w ~/LF_View_Synthesis/Inviwo/iso_depth.inv (assuming this repository was cloned to the home directory)
3. Ensure that the volume source processor is reading from the correct location. If not, change it to ... *DECIDE VOLUME*
4. The dataset can be previewed with ./inviwo -w ~/LF_View_Synthesis/Inviwo/iso_depth.inv -p ~/LF_View_Synthesis/Inviwo/LF_dataset_preview.py *TODO*
5. Or captured with ./inviwo -w ~/LF_View_Synthesis/Inviwo/iso_depth.inv -p ~/LF_View_Synthesis/Inviwo/LF_dataset_capture.py which will by default produce 50 light field images into a sub-directory of your home directory.
6. Note that the camera is moved randomly in the scripts, so in the final branch of this repo, the seed will be fixed to that used in the paper *Make sure to do*

# Replicating image warping

# Replicating Convolutional Neural Network
