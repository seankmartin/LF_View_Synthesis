#!/bin/bash

mkdir -p central_val

echo "Getting all central images"

for dir in val/*/
do
	mkdir -p "central_val/$dir"
	name="$dir""Colour44.png"
	if [ -e "$name" ]
	then
		echo "Copying $name"
		cp "$name" "central_val/$name"
	else
		echo "Couldn't find $name"
	fi
	name="$dir""Depth44.npy"
	if [ -e "$name" ]
        then
                echo "Copying $name"
                cp "$name" "central_val/$name"
        else
                echo "Couldn't find $name"
        fi
done
