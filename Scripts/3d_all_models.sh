#!/bin/bash
printf "The first variable should be the number of the epochs to test each model for\n"

cd ../
#Try the 3D model with no warping
printf "\nTesting the 3D model without geometrical warping\n"
cd Direct_3D/Scripts
./run_models.sh "--tag Test3DDirect --n $1 --cfg small.ini"
cd ../../

#Try the 3D model with warping
printf "\nTesting the 3D model with geometrical warping\n"
cd 3D/Scripts
./run_models.sh "--tag Test3D --n $1 --cfg small.ini"
cd ../../

printf "\nAll models tested\n"
