#include "util.hpp"

#include <opencv2/core.hpp>  // minMaxLoc

void rld::imgGetMinMaxVal(cv::Mat img, double &minVal, double &maxVal) {
    cv::Point minLoc;
    cv::Point maxLoc;
    cv::minMaxLoc(img, &minVal, &maxVal, &minLoc, &maxLoc);
}