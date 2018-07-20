#!/bin/bash
cd ../PythonCode
mkdir -p logs
for arg in "$@"
do
    (../Scripts/model.sh "$arg") | (tee "logs/$(date +"%FT%T").log")
done

cd ../Scripts
