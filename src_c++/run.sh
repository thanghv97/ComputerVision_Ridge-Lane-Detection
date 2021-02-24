#!/bin/bash

# export settings
. .settings

PRJ_DIR=`pwd`

BIN_DIR=$PRJ_DIR/bin
CFG_DIR=$PRJ_DIR/etc
TEST_INPUT_IMAGE_DIR=$PRJ_DIR/../test/input/image/
TEST_INPUT_VIDEO_DIR=$PRJ_DIR/../test/input/video/
TEST_OUTPUT_DIR=$PRJ_DIR/../test/output/

# TEST OPTION:
#       0 for image to image
#       1 for images to images
#       2 for images to video
#       3 for video to images 
#       4 for video to video
TEST_OPTION=0

PROG=lane_detection

if [ $TEST_OPTION -eq 0 ]
then
    INPUT_URI=$TEST_INPUT_IMAGE_DIR/input/MKZ_LONGCAMERA/1800\(US_001\).jpg
    OUTPUT_URL=$TEST_OUTPUT_DIR/rs.jpg
    
    $BIN_DIR/$PROG $CFG_DIR $TEST_OPTION $INPUT_URI $OUTPUT_URL

elif [ $TEST_OPTION -eq 1 ]
then
    INPUT_URL=$TEST_INPUT_IMAGE_DIR/ramp/
    OUTPUT_URL=$TEST_OUTPUT_DIR
    
    $BIN_DIR/$PROG $CFG_DIR $TEST_OPTION $INPUT_URL $OUTPUT_URL

elif [ $TEST_OPTION -eq 2 ]
then
    INPUT_URL=$TEST_INPUT_IMAGE_DIR/ramp/
    OUTPUT_URI=$TEST_OUTPUT_DIR/ramp.avi

    $BIN_DIR/$PROG $CFG_DIR $TEST_OPTION $INPUT_URL $OUTPUT_URI

elif [ $TEST_OPTION -eq 3 ]
then
    INPUT_URI=/home/thanghv7/Autopilot/Video/out_short3.mp4
    OUTPUT_URL=$TEST_OUTPUT_DIR
    FRAME_PER_IMAGE=30

    $BIN_DIR/$PROG $CFG_DIR $TEST_OPTION $INPUT_URI $OUTPUT_URL $FRAME_PER_IMAGE

elif [ $TEST_OPTION -eq 4 ]
then
    INPUT_URI=/home/thanghv7/Autopilot/Video/MKZ_SHORTCAMERA/2_VN_001.mp4
    OUTPUT_URI=/home/thanghv7/Autopilot/Video/MKZ_SHORTCAMERA/result/2_VN_001.avi
  
    $BIN_DIR/$PROG $CFG_DIR $TEST_OPTION $INPUT_URI $OUTPUT_URI

fi