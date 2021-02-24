#ifndef RLD_INCLUDE_HPP
#define RLD_INCLUDE_HPP

#include <iostream>
using namespace std;

#include "./platform/camera.hpp"
#include "./platform/config.hpp"
#include "./util/io.hpp"
#include "./util/log.hpp"
#include "./util/util.hpp"

#ifdef USE_CUDA
#include "./platform/gpu.hpp"
#endif

/***** CPU *****/
#include <opencv2/videoio/videoio_c.h>  // CV_CAP_PROP_FRAME_WIDTH, CV_CAP_PROP_FRAME_HEIGHT, CV_CAP_PROP_FPS

#include <opencv2/core.hpp>        // meanStdDev, sum
#include <opencv2/core/mat.hpp>    // Mat
#include <opencv2/core/types.hpp>  // Scalar, Size
#include <opencv2/imgcodecs.hpp>   // ImreadModes
#include <opencv2/imgproc.hpp>     // cvtColor, resize, gaussianBlur, calcHist, InterpolationFlags

/***** CUDA GPU *****/
#ifdef USE_CUDA
#include <opencv2/core/cuda.hpp>    // getCudaEnabledDeviceCount, getDevice, GpuMat
#include <opencv2/cudaarithm.hpp>   // multiply
#include <opencv2/cudaimgproc.hpp>  // cvtColor
#include <opencv2/cudawarping.hpp>  // resize
#endif

#endif