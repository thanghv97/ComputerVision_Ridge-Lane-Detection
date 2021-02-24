#include "gpu.hpp"

#include "include.hpp"
#include "smooth.hpp"

extern rld::Gaussian gaussian;

rld::Gpu gpu = rld::Gpu();

rld::Gpu::Gpu() {
}

rld::Gpu::~Gpu() {
}

void rld::Gpu::init(int rows, int cols) {
    /*** Init tmp image ***/
    for (int i = 0; i < 5; i++) {
        this->tmpImg[i] = cv::cuda::GpuMat(rows, cols, CV_32F);
    }

    /*** Init kernel gaussian ***/
    this->gausKernelSizeX = cv::cuda::GpuMat(1, rows, CV_8U);
    this->gausKernelSizeY = cv::cuda::GpuMat(1, rows, CV_8U);
    this->gausKernelTensorSizeX = cv::cuda::GpuMat(1, rows, CV_8U);
    this->gausKernelTensorSizeY = cv::cuda::GpuMat(1, rows, CV_8U);

    cv::Mat kSizeTmpX(1, rows, CV_8U);
    cv::Mat kSizeTmpY(1, rows, CV_8U);
    cv::Mat kTensorSizeTmpX(1, rows, CV_8U);
    cv::Mat kTensorSizeTmpY(1, rows, CV_8U);

    for (int i = 0; i < rows; i++) {
        // Kernel X
        cv::Mat kernel = cv::getGaussianKernel(gaussian.gausRatio[0].at<float>(i, 0), gaussian.gausRatio[1].at<float>(i, 0), CV_32F);
        this->gausKernelX.push_back(cv::cuda::GpuMat(kernel.cols, kernel.rows, kernel.type()));
        this->gausKernelX[i].upload(kernel.t());
        kSizeTmpX.at<uchar>(0, i) = kernel.rows;

        // kernel Y
        kernel = cv::getGaussianKernel(gaussian.gausRatio[0].at<float>(i, 0), gaussian.gausRatio[2].at<float>(i, 0), CV_32F);
        this->gausKernelY.push_back(cv::cuda::GpuMat(kernel.cols, kernel.rows, kernel.type()));
        this->gausKernelY[i].upload(kernel.t());
        kSizeTmpY.at<uchar>(0, i) = kernel.rows;

        // kernel tensor X
        kernel = cv::getGaussianKernel(gaussian.gausRatioTensor[0].at<float>(i, 0), gaussian.gausRatioTensor[1].at<float>(i, 0), CV_32F);
        this->gausKernelTensorX.push_back(cv::cuda::GpuMat(kernel.cols, kernel.rows, kernel.type()));
        this->gausKernelTensorX[i].upload(kernel.t());
        kTensorSizeTmpX.at<uchar>(0, i) = kernel.rows;

        // kernel tensor Y
        kernel = cv::getGaussianKernel(gaussian.gausRatioTensor[0].at<float>(i, 0), gaussian.gausRatioTensor[2].at<float>(i, 0), CV_32F);
        this->gausKernelTensorY.push_back(cv::cuda::GpuMat(kernel.cols, kernel.rows, kernel.type()));
        this->gausKernelTensorY[i].upload(kernel.t());
        kTensorSizeTmpY.at<uchar>(0, i) = kernel.rows;
    }
    this->gausKernelSizeX.upload(kSizeTmpX);
    this->gausKernelSizeY.upload(kSizeTmpY);
    this->gausKernelTensorSizeX.upload(kTensorSizeTmpX);
    this->gausKernelTensorSizeY.upload(kTensorSizeTmpY);

    /*** init gradient filter ***/
    this->filterSobelX = cv::cuda::createSobelFilter(CV_32F, CV_32F, 1, 0, 3, (double)1 / 128);
    this->filterSobelY = cv::cuda::createSobelFilter(CV_32F, CV_32F, 0, 1, 3, (double)1 / 128);

    cv::Mat kernX = (cv::Mat_<float>(1, 3) << -0.5, 0, 0.5);
    cv::Mat kernY = (cv::Mat_<float>(3, 1) << -0.5, 0, 0.5);
    this->filterLinearX = cv::cuda::createLinearFilter(CV_32F, CV_32F, kernX, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
    this->filterLinearY = cv::cuda::createLinearFilter(CV_32F, CV_32F, kernY, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);

    /*** init tensor struct ***/
    this->imgXX = cv::cuda::GpuMat(rows, cols, CV_32F);
    this->imgXY = cv::cuda::GpuMat(rows, cols, CV_32F);
    this->imgYY = cv::cuda::GpuMat(rows, cols, CV_32F);

    this->eigenVals = {cv::cuda::GpuMat(rows, cols, CV_32F), cv::cuda::GpuMat(rows, cols, CV_32F)};
    this->domintVtr = {cv::cuda::GpuMat(rows, cols, CV_32F), cv::cuda::GpuMat(rows, cols, CV_32F)};
    this->grad = {cv::cuda::GpuMat(rows, cols, CV_32F), cv::cuda::GpuMat(rows, cols, CV_32F)};
}