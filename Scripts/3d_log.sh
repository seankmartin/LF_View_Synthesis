#!/bin/bash

mkdir -p logs
(./3d_all_models.sh "$1") | (tee "./logs/$(date +"%FT%T").log")
