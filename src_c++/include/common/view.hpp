#ifndef RLD_VIEW_HPP
#define RLD_VIEW_HPP

#include <opencv2/core/mat.hpp>

namespace rld {

struct birdViewCfg {
    int u0;
    int v0;
    int skew;
    float minRow;
    float maxRow;
    float minCol;
    float maxCol;
};

class View {
   private:
    cv::Mat birdViewMask;
    cv::Mat birdViewMaskInv;

   public:
    birdViewCfg bvCfg;
    View();
    ~View();
    void calBirdViewMask();
    cv::Mat getBirdViewMask();
    cv::Mat getBirdViewMaskInv();
    static void cvtCameraView2BirdView(cv::Mat srcImg, cv::Mat &dstImg);
    static void cvtBirdView2CameraView(cv::Mat srcImg, cv::Mat &dstImg);
};

}  // namespace rld

#endif