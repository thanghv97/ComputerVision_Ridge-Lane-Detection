#ifndef RLD_NORMALIZE_HPP
#define RLD_NORMALIZE_HPP

#include <opencv2/core/mat.hpp>

namespace rld {

class Normalize {
   public:
    Normalize();
    ~Normalize();
    static void imgValueNormalize(cv::Mat srcImg, cv::Mat &dstImg, int minValNor, int maxValNor);
    static void imgHistogramNormalize(cv::Mat srcImg, cv::Mat &dstImg);
};

}  // namespace rld

#endif