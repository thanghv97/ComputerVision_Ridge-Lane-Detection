#ifndef RLD_LANE_DETECTION_HPP
#define RLD_LANE_DETECTION_HPP

#include <opencv2/core/mat.hpp>

cv::Mat ridgeDetection(cv::Mat img, std::string imgPath, bool isVisualizeStepImg);

#endif