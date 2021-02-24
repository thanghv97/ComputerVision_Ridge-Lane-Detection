#ifndef RLD_KERNEL_ANISOTROPIC_GAUSSIAN_H
#define RLD_KERNEL_ANISOTROPIC_GAUSSIAN_H

#include <opencv2/core/cuda.hpp>

namespace rld {
namespace kernel {
void cudaAnisotropicGaussian(cv::cuda::GpuMat srcImg, cv::cuda::GpuMat &dstImg,
                             std::vector<cv::cuda::GpuMat> kernelX, std::vector<cv::cuda::GpuMat> kernelY,
                             cv::cuda::GpuMat kernelSizeX, cv::cuda::GpuMat kernelSizeY);
}
}  // namespace rld

#endif