#!/bin/bash
cd ../PythonCode
for arg in "$@"
do
    (../Scripts/model.sh "$arg") | (tee "$HOME/turing/output/logs/$(date +"%FT%T").log")
done

cd ../Scripts
