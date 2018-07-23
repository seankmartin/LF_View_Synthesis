#!/bin/bash
cd ..
mkdir -p logs
cd PythonCode
for arg in "$@"
do
    (../Scripts/model.sh "$arg") | (tee "../logs/$(date +"%FT%T").log")
done

cd ../Scripts
