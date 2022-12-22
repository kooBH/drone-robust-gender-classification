#!/bin/bash

DIR_IN=<directory_of_input_files>
DIR_OUT=<directory_of_output_files>

VERSION=v24 # see config directory
DEVICE=cuda:0 # or cuda:1 or cpu
python run.py --config config/${VERSION}.yaml -d ${DEVICE} -i ${DIR_IN} -o ${DIR_OUT}