#!/bin/bash

#Test different learning rate strategies
cd ..
./run_models.sh "--tag Cyclical --n 50 --s cyclical --cfg small.ini --lr 0.1" "--tag Warm --n 50 --s warm --cfg small.ini --lr 0.1" "--tag Plateau --n 50 --s step --cfg small.ini --lr 0.1"
