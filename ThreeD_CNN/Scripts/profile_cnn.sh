#!/bin/bash
cd ..
mkdir -p logs
cd PythonCode
(python3 -m torch.utils.bottleneck cnn_main.py "$1") | (tee "../logs/$(date +"%FT%T").log")
echo "$1"
cd ../Scripts
