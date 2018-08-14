#!/bin/bash

mkdir -p logs
(./test_all_models.sh "$1") | (tee "./logs/$(date +"%FT%T").log")
