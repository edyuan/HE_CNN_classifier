#!/usr/bin/env bash

. ~/anaconda2/envs/tensorflow/bin/activate tensorflow

python ./Modules/para_probe_detection_yellow.py $1 $2 $3 $4

. ~/anaconda2/envs/tensorflow/bin/deactivate
