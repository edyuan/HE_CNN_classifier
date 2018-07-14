#!/bin/bash

. ~/anaconda2/envs/tensorflow/bin/activate tensorflow

python ./Modules/para_he_dcis_segmentation.py $1 $2 $3

. ~/anaconda2/envs/tensorflow/bin/deactivate

