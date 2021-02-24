#ifndef RLD_GPU_HPP
#define RLD_GPU_HPP

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafilters.hpp>

namespace rld {

class Gpu {
   public:
    cv::cuda::GpuMat frame;
    cv::cuda::GpuMat rsImg;
    cv::cuda::GpuMat visualImg;

    cv::cuda::GpuMat tmpImg[5];

    /*** gaussian smooth ***/
    std::vector<cv::cuda::GpuMat> gausKernelX;
    std::vector<cv::cuda::GpuMat> gausKernelY;
    std::vector<cv::cuda::GpuMat> gausKernelTensorX;
    std::vector<cv::cuda::GpuMat> gausKernelTensorY;
    cv::cuda::GpuMat gausKernelSizeX;
    cv::cuda::GpuMat gausKernelSizeY;
    cv::cuda::GpuMat gausKernelTensorSizeX;
    cv::cuda::GpuMat gausKernelTensorSizeY;

    /*** gradient ***/
    cv::Ptr<cv::cuda::Filter> filterSobelX;
    cv::Ptr<cv::cuda::Filter> filterSobelY;
    cv::Ptr<cv::cuda::Filter> filterLinearX;
    cv::Ptr<cv::cuda::Filter> filterLinearY;
    std::vector<cv::cuda::GpuMat> grad;

    /*** tensor struct ***/
    cv::cuda::GpuMat imgXX, imgYY, imgXY;
    std::vector<cv::cuda::GpuMat> eigenVals;
    std::vector<cv::cuda::GpuMat> domintVtr;
    cv::cuda::GpuMat det, trace, mask;

    Gpu();
    ~Gpu();
    void init(int rows, int cols);
};

}  // namespace rld

#endif