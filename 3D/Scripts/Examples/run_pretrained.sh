#!/bin/bash

cd ..
./run_models.sh "--tag Pretrained3D --pretrained NoMultiWarp_best_at14.pth --n 10 --cfg no_multi.ini --first=False"
