#ifndef RLD_CUDA_LANE_DETECTION_HPP
#define RLD_CUDA_LANE_DETECTION_HPP

#include <opencv2/core/cuda.hpp>

cv::cuda::GpuMat ridgeDetection(cv::cuda::GpuMat img, std::string imgPath, bool isVisualizeStepImg);

#endif