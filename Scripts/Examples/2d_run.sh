#!/bin/bash

cd ../

./run_models.sh "--tag 2D_model_small --pretrained EDSR_x2.pt --n 10 --cfg small.ini --prompt"
