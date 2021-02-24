#include "cuda_lane_detection.hpp"

#include <chrono>
#include <thread>

#include "common/normalize.hpp"
#include "common/shadow_removal.hpp"
#include "common/smooth.hpp"
#include "filter_mechanism.hpp"
#include "include.hpp"
#include "kernel/dominant_vector.h"

extern rld::Camera camera;
extern rld::Config config;
extern rld::Gaussian gaussian;
extern rld::Gpu gpu;

void executeGpuByCpuFunc(cv::cuda::GpuMat srcImgGpu, cv::cuda::GpuMat &dstImgGpu,
                         void (*func)(cv::Mat srcImgCpu, cv::Mat &dstImgCpu)) {
    cv::Mat tmpImg;
    srcImgGpu.download(tmpImg);
    (*func)(tmpImg, tmpImg);
    dstImgGpu.upload(tmpImg);
}

void executeGpuByCpuFunc(cv::cuda::GpuMat srcImgGpu, cv::cuda::GpuMat &dstImgGpu, std::vector<cv::cuda::GpuMat> paramGpu,
                         void (*func)(cv::Mat srcImgCpu, cv::Mat &dstImgCpu, std::vector<cv::Mat> paramCpu)) {
    std::vector<cv::Mat> tmpParam;
    cv::Mat tmpImg;
    srcImgGpu.download(tmpImg);
    for (uint8_t i = 0; i < paramGpu.size(); i++) {
        paramGpu[i].download(tmpParam[i]);
    }
    (*func)(tmpImg, tmpImg, tmpParam);
    dstImgGpu.upload(tmpImg);
}

/**************************************************************************************************
 *                                                                                                *
 *                                     PRE_PROCESS_ALGORITHM                                      *
 *                                                                                                *
 **************************************************************************************************/
void convert2GrayImg(cv::cuda::GpuMat srcImg, cv::cuda::GpuMat &dstImg) {
    int opt = 0;

    switch (opt) {
        case 0: /*** gray by opencv ***/
            cv::cuda::cvtColor(srcImg, dstImg, cv::COLOR_BGR2GRAY);
            break;
        case 1: /*** gray by illuminant invariance ***/
            executeGpuByCpuFunc(srcImg, dstImg, rld::ShadowRemoval::imgIlluminantInvariance);
            break;
        default:
            break;
    }
}

void normalizeGrayImg(cv::cuda::GpuMat srcImg, cv::cuda::GpuMat &dstImg) {
    int opt = -1;

    switch (opt) {
        case 0: /*** normalize by value ***/
            // executeGpuByCpuFunc(srcImg, dstImg, rld::Normalize::imgValueNormalize);
            break;
        case 1: /*** normalize by histogram ***/
            executeGpuByCpuFunc(srcImg, dstImg, rld::Normalize::imgHistogramNormalize);
            break;
        default:
            break;
    }
}

static void preProcessAlgorithm(cv::cuda::GpuMat srcImg, cv::cuda::GpuMat &dstImg, string imgPath, bool isLogImg) {
    /***** Convert to gray input image *****/
    convert2GrayImg(srcImg, srcImg);
    rld::imwrite(srcImg, imgPath + "/01_01_Gray_Image.jpg", isLogImg);

    /***** Normalize gray image *****/
    normalizeGrayImg(srcImg, srcImg);
    rld::imwrite(srcImg, imgPath + "/01_02_Gray_Image_Normalize.jpg", isLogImg);

    srcImg.convertTo(dstImg, CV_32F);
}

/**************************************************************************************************
 *                                                                                                *
 *                                     PROCESS_ALGORITHM                                          *
 *                                                                                                *
 **************************************************************************************************/
static void computeDerivatives(cv::cuda::GpuMat img) {
    /*** gradient by sobel - opencv ***/
    gpu.filterSobelX->apply(img, gpu.grad[0]);
    gpu.filterSobelY->apply(img, gpu.grad[1]);
}

static void computeTensorStruct() {
    cv::cuda::multiply(gpu.grad[0], gpu.grad[0], gpu.imgXX);
    cv::cuda::multiply(gpu.grad[0], gpu.grad[1], gpu.imgXY);
    cv::cuda::multiply(gpu.grad[1], gpu.grad[1], gpu.imgYY);

    gaussian.imgAnisotropicGaussianSmoothByBirdView(gpu.imgXX, gpu.imgXX, rld::GAUSSIAN_KERNEL_TENSOR);
    gaussian.imgAnisotropicGaussianSmoothByBirdView(gpu.imgXY, gpu.imgXY, rld::GAUSSIAN_KERNEL_TENSOR);
    gaussian.imgAnisotropicGaussianSmoothByBirdView(gpu.imgYY, gpu.imgYY, rld::GAUSSIAN_KERNEL_TENSOR);
}

void computeDivergence() {
    gpu.filterLinearX->apply(gpu.domintVtr[0], gpu.tmpImg[0]);
    gpu.filterLinearY->apply(gpu.domintVtr[1], gpu.tmpImg[1]);

    cv::cuda::add(gpu.tmpImg[0], gpu.tmpImg[1], gpu.tmpImg[2]);
    cv::cuda::multiply(gpu.tmpImg[2], -1, gpu.rsImg);
}

cv::cuda::GpuMat ridgeDetection(cv::cuda::GpuMat img, string imgPath, bool isVisualizeStepImg) {
    rld::imwrite(img, imgPath + "/00_Origin_Image.jpg", isVisualizeStepImg);
    /******************** PROCESS ********************/
    /***** Pre-process *****/
    preProcessAlgorithm(img, img, imgPath, isVisualizeStepImg);

    /*** Step 1: Gaussian smoothing ***/
    gaussian.imgAnisotropicGaussianSmoothByBirdView(img, img, rld::GAUSSIAN_KERNEL_COMMON);
    rld::imwrite(img, imgPath + "/02_Smooth_Image.jpg", isVisualizeStepImg);

    /***  Step 2: Compute derivatives => gradient vector field ***/
    computeDerivatives(img);
    rld::imwrite(gpu.grad[0], imgPath + "/03_01_Gradient_X.jpg", isVisualizeStepImg);
    rld::imwrite(gpu.grad[1], imgPath + "/03_02_Gradient_Y.jpg", isVisualizeStepImg);

    /*** Step 3: Build structure tensor ***/
    computeTensorStruct();

    /*** Step 4: Get dominant vector ***/
    rld::kernel::cudaDominantVector();

    /*** Step 5: Compute divergence ***/
    computeDivergence();
    cv::cuda::compare(gpu.rsImg, 0.25f, gpu.mask, CV_CMP_LT);
    gpu.rsImg.setTo(cv::Scalar::all(0), gpu.mask);
    cv::cuda::multiply(gpu.rsImg, 128.0, gpu.visualImg);
    rld::imwrite(gpu.visualImg, imgPath + "/04_Origin_Ridge_Image.jpg", isVisualizeStepImg);

    /*** Step 6: Apply ridge filter ***/
    /* Noise filter */
    if (IS_USING_RIDGE_NOISE_FILTER) {
        if (config.common.isVisualizeFilterImg) {
        }
        executeGpuByCpuFunc(gpu.rsImg, gpu.rsImg, ridgeNoiseFilter);
        cv::cuda::multiply(gpu.rsImg, 128.0, gpu.visualImg);
        rld::imwrite(gpu.visualImg, imgPath + "/05_02_Filter_Noise_Ridge_Image.jpg", isVisualizeStepImg);
    }

    /* Theta filter: Discard ridge point with large horizontal component */
    if (IS_USING_RIDGE_THETA_FILTER) {
        if (config.common.isVisualizeFilterImg) {
            // cv::Mat visualImage = visualizeImageRidgeThetaFilter(ridgeImg, dominantVt);
            // rld::imwrite(visualImage, imgPath + "/05_01_Visualize_Theta_Ridge.jpg", isVisualizeStepImg);
        }
        executeGpuByCpuFunc(gpu.rsImg, gpu.rsImg, gpu.domintVtr, ridgeThetaFilter);
        cv::cuda::multiply(gpu.rsImg, 128.0, gpu.visualImg);
        rld::imwrite(gpu.visualImg, imgPath + "/05_01_Filter_Theta_Ridge_Image.jpg", isVisualizeStepImg);
    }

    /* Confident filter */
    if (IS_USING_RIDGE_CONFIDENCE_FILTER) {
        if (config.common.isVisualizeFilterImg) {
            // cv::Mat visualImage = visualizeImageRidgeConfidenceFilter(ridgeImg, eigenValue);
            // rld::imwrite(visualImage, imgPath + "/05_03_Visualize_Confidence_Ridge.jpg", isVisualizeStepImg);
        }
        ridgeConfidenceFilter(gpu.rsImg, gpu.rsImg, gpu.eigenVals);
        cv::cuda::multiply(gpu.rsImg, 128.0, gpu.visualImg);
        rld::imwrite(gpu.visualImg, imgPath + "/05_03_Filter_Confidence_Ridge_Image.jpg", isVisualizeStepImg);
    }

    cv::cuda::multiply(gpu.rsImg, 128.0, gpu.rsImg);
    return gpu.rsImg;
}