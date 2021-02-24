#ifndef RLD_SMOOTH_HPP
#define RLD_SMOOTH_HPP

#include <opencv2/core/mat.hpp>

#ifdef USE_CUDA
#include <opencv2/core/cuda.hpp>
#endif

namespace rld {

enum gaussianKernelType {
    GAUSSIAN_KERNEL_COMMON = 0,
    GAUSSIAN_KERNEL_TENSOR
};

struct gaussianKernelCfg {
    int kSize;
    float sigmaX;
    float sigmaY;
};

class Gaussian {
   public:
    gaussianKernelCfg gausKCommon;
    gaussianKernelCfg gausKTensor;
    std::vector<cv::Mat> gausRatio;
    std::vector<cv::Mat> gausRatioTensor;
    Gaussian();
    ~Gaussian();
    void calGaussianRatio(int rows, int cols, rld::gaussianKernelType type, std::vector<cv::Mat> &gausRatio);
    void getGaussianCfg(int &kSize, float &sigmaX, float &sigmaY, rld::gaussianKernelType type);
    static void imgGaussianSmooth(cv::Mat srcImg, cv::Mat &dstImg, rld::gaussianKernelType type);
    static void imgAnisotropicGaussianSmooth(cv::Mat srcImg, cv::Mat &dstImg, rld::gaussianKernelType type);
    static void imgAnisotropicGaussianSmoothByBirdView(cv::Mat srcImg, cv::Mat &dstImg, rld::gaussianKernelType type);

#ifdef USE_CUDA
    static void imgAnisotropicGaussianSmoothByBirdView(cv::cuda::GpuMat srcImg, cv::cuda::GpuMat &tmpImg, rld::gaussianKernelType type);
#endif
};

}  // namespace rld

#endif