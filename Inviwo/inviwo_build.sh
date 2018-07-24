#!/bin/bash

if ["$1" == ""]
then
	echo "Please enter the path to the relevant qt cmake folder"
else
	export CMAKE_PREFIX_PATH="$1"
	echo $CMAKE_PREFIX_PATH
	mkdir -p inviwo_build
	cd inviwo_build
	cmake -G "Unix Makefiles" -D IVW_MODULE_DEPTH=1 ../inviwo
	#8 indicates the number of cores to build with
	make -j8
	echo "Starting Inviwo"
	./bin/inviwo
fi
