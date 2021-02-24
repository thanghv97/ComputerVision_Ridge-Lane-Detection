#!/bin/bash
PRJ_DIR=`pwd`

PROG=./get_camera_params

echo "test birdview parameters"
INPUT_URI=$PRJ_DIR/../../test/input/image/MKZ_SHORTCAMERA/18.jpg
# INPUT_URI=/home/thanghv7/Autopilot/Source/ridge-lane-detection/src_c++/image/output/ridge_image.jpg
SCALE_RESIZE=0.5

$PROG $INPUT_URI $SCALE_RESIZE