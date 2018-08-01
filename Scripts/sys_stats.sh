#!/bin/bash

printf "Python processes:\n"
./list_python_processes.sh
printf "\nShared memory:\n"
./list_shared_mem.sh
printf "\nRAM:\n"
free -m
printf "\nCPU memory:\n"
cat /proc/meminfo
