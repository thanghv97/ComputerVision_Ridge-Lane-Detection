#ifndef RLD_FILTER_MECHANISM_HPP
#define RLD_FILTER_MECHANISM_HPP

#include <opencv2/core/mat.hpp>

#ifdef USE_CUDA
#include <opencv2/core/cuda.hpp>
#endif

#define IS_USING_RIDGE_THETA_FILTER 0
#define IS_USING_RIDGE_NOISE_FILTER 0
#define IS_USING_RIDGE_CONFIDENCE_FILTER 1

void ridgeThetaFilter(cv::Mat srcImg, cv::Mat &dstImg, std::vector<cv::Mat> dominantVt);
void ridgeConfidenceFilter(cv::Mat srcImg, cv::Mat &dstImg, const std::vector<cv::Mat> eigenValue);
void ridgeNoiseFilter(cv::Mat srcImg, cv::Mat &dstImg);

cv::Mat visualizeImageRidgeThetaFilter(const cv::Mat srcImg, const std::vector<cv::Mat> dominantVector);
cv::Mat visualizeImageRidgeConfidenceFilter(const cv::Mat srcImg, const std::vector<cv::Mat> eigenValue);

#ifdef USE_CUDA
void ridgeConfidenceFilter(cv::cuda::GpuMat srcImg, cv::cuda::GpuMat &dstImg, const std::vector<cv::cuda::GpuMat> eigenValue);
#endif

#endif