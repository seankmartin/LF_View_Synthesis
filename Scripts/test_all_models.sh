#!/bin/bash
echo "The first variable should be the number of the epochs to test each model for"

#Try the 2D without pretrained
echo "Testing the 2D model with no pretrained data"
cd ../2D/Scripts
./run_models.sh "--tag Test2D --n $1 --cfg small.ini"
cd ../../

#Try the 2D model with pretrained
echo "Testing the 2D model with pretrained data"
cd Pretrained_2D/Scripts
./run_models.sh "--tag Test2DPre --pretrained EDSR_x2.pt --n $1 --cfg small.ini"
cd ../../

#Try the 3D model with no warping
echo "Testing the 3D model without geometrical warping"
cd Direct_3D/Scripts
./run_models.sh "--tag Test3DDirect --n $1 --cfg small.ini"
cd ../../

#Try the 3D model with warping
echo "Testing the 3D model with geometrical warping"
cd Direct_3D/Scripts
./run_models.sh "--tag Test3D --n $1 --cfg small.ini"
cd ../../

echo "All models tested"
