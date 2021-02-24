#include "smooth.hpp"

#include <cmath>

#include "anisotropic_gaussian.h"
#include "include.hpp"
#include "view.hpp"

extern rld::View view;

#ifdef USE_CUDA
extern rld::Gpu gpu;
#endif 

rld::Gaussian gaussian = rld::Gaussian();

rld::Gaussian::Gaussian() {
}

rld::Gaussian::~Gaussian() {
}

void rld::Gaussian::getGaussianCfg(int &kSize, float &sigmaX, float &sigmaY, rld::gaussianKernelType type) {
    switch (type) {
        case rld::GAUSSIAN_KERNEL_COMMON:
            kSize = this->gausKCommon.kSize;
            sigmaX = this->gausKCommon.sigmaX;
            sigmaY = this->gausKCommon.sigmaY;
            break;
        case rld::GAUSSIAN_KERNEL_TENSOR:
            kSize = this->gausKTensor.kSize;
            sigmaX = this->gausKTensor.sigmaX;
            sigmaY = this->gausKTensor.sigmaY;
            break;
        default:
            break;
    }
}

inline static void calDistRatio(const cv::Mat point, const cv::Mat point1, const cv::Mat point2,
                                const int sizeX, const int sizeY, float &distX, float &distY, float &projectedY) {
    cv::Mat bvPoint = view.getBirdViewMaskInv() * point;
    cv::Mat bvPoint1 = view.getBirdViewMaskInv() * point1;
    cv::Mat bvPoint2 = view.getBirdViewMaskInv() * point2;

    bvPoint = bvPoint / bvPoint.at<float>(2, 0);
    bvPoint1 = bvPoint1 / bvPoint1.at<float>(2, 0);
    bvPoint2 = bvPoint2 / bvPoint2.at<float>(2, 0);
    distX = fabs(bvPoint.at<float>(0, 0) - bvPoint1.at<float>(0, 0));
    distX /= sizeX;
    distY = fabs(bvPoint.at<float>(1, 0) - bvPoint2.at<float>(1, 0));
    distY /= sizeY;
    projectedY = bvPoint.at<float>(1, 0);
}

void rld::Gaussian::calGaussianRatio(int rows, int cols, rld::gaussianKernelType type, std::vector<cv::Mat> &gausRatio) {
    int kSize = 0;
    float sigmaX = 0, sigmaY = 0;
    gausRatio = {cv::Mat(rows, 1, CV_32F), cv::Mat(rows, 1, CV_32F), cv::Mat(rows, 1, CV_32F)};

    /******************** PROCESS ********************/
    /***** Get config of type kernel *****/
    getGaussianCfg(kSize, sigmaX, sigmaY, type);
    if (kSize == 0) {
        cout << "ERROR: Not found config of gaussian kernel (type[" << type << "] !!!" << endl;
        return;
    }

    /***** Get baseDistX, baseDistY *****/
    cv::Mat point = (cv::Mat_<float>(3, 1) << cols - 1, rows, 1);
    cv::Mat point1 = (cv::Mat_<float>(3, 1) << 0, rows, 1);
    cv::Mat point2 = (cv::Mat_<float>(3, 1) << cols - 1, rows - 10, 1);

    float baseDistX, baseDistY, baseProjectedY;
    calDistRatio(point, point1, point2, cols, 10, baseDistX, baseDistY, baseProjectedY);

    for (int i = 0; i < rows; i++) {
        /***** Get currDistX, currDistY *****/
        point = (cv::Mat_<float>(3, 1) << cols - 1, i, 1);
        point1 = (cv::Mat_<float>(3, 1) << 0, i, 1);
        if (i < rows - 10) {
            point2 = (cv::Mat_<float>(3, 1) << cols - 1, i + 10, 1);
        } else {
            point2 = (cv::Mat_<float>(3, 1) << cols - 1, i - 10, 1);
        }
        float currDistX, currDistY, currProjectedY;
        calDistRatio(point, point1, point2, cols, 10, currDistX, currDistY, currProjectedY);

        // if (currProjectedY> baseProjectedY){
        //      gausRatio[0].at<float>(i, 0) = 0;
        //      continue;
        // }
        /***** Gaussian blur by ratio currDist and baseDist *****/
        float ratioX = currDistX / baseDistX;
        float ratioY = currDistY / baseDistY;

        gausRatio[0].at<float>(i, 0) = kSize / ratioX;
        gausRatio[1].at<float>(i, 0) = sigmaX / ratioX;
        gausRatio[2].at<float>(i, 0) = sigmaY / ratioY;

        if (gausRatio[0].at<float>(i, 0) < 3) {
            gausRatio[0].at<float>(i, 0) = 3;
            gausRatio[1].at<float>(i, 0) = 1;
            gausRatio[2].at<float>(i, 0) = 1;
        }
        int currKSize = ceil(gausRatio[0].at<float>(i, 0));
        if (currKSize % 2 == 0) {
            currKSize -= 1;
        }
        gausRatio[0].at<float>(i, 0) = currKSize;
    }
}

void rld::Gaussian::imgGaussianSmooth(cv::Mat srcImg, cv::Mat &dstImg, rld::gaussianKernelType type) {
    int kSize = 0;
    float sigmaX = 0, sigmaY = 0;

    /******************** PROCESS ********************/
    gaussian.getGaussianCfg(kSize, sigmaX, sigmaY, type);
    if (kSize == 0) {
        cout << "ERROR: Not found config of gaussian kernel (type[" << type << "] !!!" << endl;
        return;
    }
    cv::GaussianBlur(srcImg, dstImg, cv::Size(kSize, kSize), sigmaX, sigmaY);
}

void rld::Gaussian::imgAnisotropicGaussianSmooth(cv::Mat srcImg, cv::Mat &dstImg, rld::gaussianKernelType type) {
    int rows = srcImg.rows;
    int cols = srcImg.cols;
    int kSize = 0;
    float sigmaX = 0, sigmaY = 0;

    /******************** PROCESS ********************/
    /***** Get config of type kernel *****/
    gaussian.getGaussianCfg(kSize, sigmaX, sigmaY, type);
    if (kSize == 0) {
        cout << "ERROR: Not found config of gaussian kernel (type[" << type << "] !!!" << endl;
        return;
    }

    /***** Set parameter slide *****/
    float sigmaBottomX = sigmaX;
    float sigmaVanishX = 0.5;
    int n = 12;

    int rowMargin = rows / n;
    float sigmaMargin = (sigmaBottomX - sigmaVanishX) / n;

    /***** Set currrent sigma per step *****/
    float currSigmaX = sigmaBottomX;
    float currSigmaY = sigmaY;
    int currKSize = kSize;
    for (int i = n; i > 0; i--) {
        cv::Mat cropImg = srcImg.colRange(0, cols).rowRange((i - 1) * rowMargin, i * rowMargin);
        cv::GaussianBlur(cropImg, cropImg, cv::Size(currKSize, currKSize), currSigmaX, currSigmaY);
        currSigmaX -= sigmaMargin;
    }
    srcImg.copyTo(dstImg);
}

void rld::Gaussian::imgAnisotropicGaussianSmoothByBirdView(cv::Mat srcImg, cv::Mat &dstImg, rld::gaussianKernelType type) {
    std::vector<cv::Mat> gausRatio;
    if (type == rld::GAUSSIAN_KERNEL_COMMON) {
        gausRatio = gaussian.gausRatio;
    } else if (type == rld::GAUSSIAN_KERNEL_TENSOR) {
        gausRatio = gaussian.gausRatioTensor;
    }

    for (int i = 0; i < srcImg.rows; i++) {
        cv::Mat cropImg = srcImg.colRange(0, srcImg.cols).rowRange(i, i + 1);
        cv::GaussianBlur(cropImg, cropImg, cv::Size(gausRatio[0].at<float>(i, 0), gausRatio[0].at<float>(i, 0)),
                         gausRatio[1].at<float>(i, 0), gausRatio[2].at<float>(i, 0));
    }
    srcImg.copyTo(dstImg);
}

#ifdef USE_CUDA
void rld::Gaussian::imgAnisotropicGaussianSmoothByBirdView(cv::cuda::GpuMat srcImg, cv::cuda::GpuMat &dstImg, rld::gaussianKernelType type) {
    if (type == rld::GAUSSIAN_KERNEL_COMMON) {
        rld::kernel::cudaAnisotropicGaussian(srcImg, dstImg, gpu.gausKernelX, gpu.gausKernelY,
                                             gpu.gausKernelSizeX, gpu.gausKernelSizeY);
    } else if (type == rld::GAUSSIAN_KERNEL_TENSOR) {
        rld::kernel::cudaAnisotropicGaussian(srcImg, dstImg, gpu.gausKernelTensorX, gpu.gausKernelTensorY,
                                             gpu.gausKernelTensorSizeX, gpu.gausKernelTensorSizeY);
    }
}
#endif