#ifndef RLD_SHADOW_REMOVAL_HPP
#define RLD_SHADOW_REMOVAL_HPP

#include <opencv2/core/mat.hpp>

namespace rld {

class ShadowRemoval {
   public:
    ShadowRemoval();
    ~ShadowRemoval();
    static std::pair<int, std::vector<cv::Mat>> derive1DShadowFreeImage(cv::Mat srcImg, cv::Mat &dstImg);
    static void derive2DShadowFreeImage(cv::Mat srcImg, cv::Mat &dstImg, const std::vector<cv::Mat> xPlane, const int theta);
    static void imgIlluminantInvariance(cv::Mat srcImg, cv::Mat &dstImg);
};

}  // namespace rld

#endif