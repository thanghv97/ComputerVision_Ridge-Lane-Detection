#include "lane_detection.hpp"

#include <chrono>
#include <thread>

#include "Eigen/Dense"  // Matrix
#include "filter_mechanism.hpp"
#include "include.hpp"
#include "normalize.hpp"
#include "shadow_removal.hpp"
#include "smooth.hpp"

extern rld::Camera camera;
extern rld::Config config;
extern rld::Gaussian gaussian;

/**************************************************************************************************
 *                                                                                                *
 *                                     PRE_PROCESS_ALGORITHM                                      *
 *                                                                                                *
 **************************************************************************************************/
void convert2GrayImg(cv::Mat srcImg, cv::Mat &dstImg) {
    std::vector<cv::Mat> chnImg;
    int opt = 0;

    switch (opt) {
        case 0: /*** gray by opencv ***/
            cv::cvtColor(srcImg, dstImg, cv::COLOR_BGR2GRAY);
            break;
        case 1: /*** gray by illuminant invariance ***/
            // rld::Gaussian::imgGaussianSmooth(srcImg, srcImg, 5, 0.5, 0.5);
            rld::ShadowRemoval::derive1DShadowFreeImage(srcImg, dstImg);
            break;
        case 2: /*** gray by equation ***/
            cv::split(srcImg, chnImg);
            dstImg = 0.0722 * chnImg[0] + 0.7152 * chnImg[1] + 0.2126 * chnImg[2];
        default:
            break;
    }
}

void normalizeGrayImg(cv::Mat srcImg, cv::Mat &dstImg) {
    int opt = -1;

    switch (opt) {
        case 0: /*** normalize by value ***/
            rld::Normalize::imgValueNormalize(srcImg, dstImg, 0, 255);
            break;
        case 1: /*** normalize by histogram ***/
            rld::Normalize::imgHistogramNormalize(srcImg, dstImg);
            break;
        default:
            break;
    }
}

static void preProcessAlgorithm(cv::Mat srcImg, cv::Mat &dstImg, string imgPath, bool isLogImg) {
    /***** Convert to gray input image *****/
    convert2GrayImg(srcImg, srcImg);
    rld::imwrite(srcImg, imgPath + "/01_01_Gray_Image.jpg", isLogImg);

    /***** Normalize gray image *****/
    normalizeGrayImg(srcImg, srcImg);
    rld::imwrite(srcImg, imgPath + "/01_02_Gray_Image_Normalize.jpg", isLogImg);

    srcImg.convertTo(srcImg, CV_32F);
    srcImg.copyTo(dstImg);
}

/**************************************************************************************************
 *                                                                                                *
 *                                     PRE_PROCESS_ALGORITHM                                      *
 *                                                                                                *
 **************************************************************************************************/
static std::vector<cv::Mat> computeDerivatives(cv::Mat img) {
    cv::Mat gradX, gradY;

    /*** gradient by sobel - opencv ***/
    cv::Sobel(img, gradX, CV_32F, 1, 0, 3, (double)1 / 128);
    cv::Sobel(img, gradY, CV_32F, 0, 1, 3, (double)1 / 128);
    return {gradX, gradY};
}

static void computeTensorStruct(const std::vector<cv::Mat> grad, cv::Mat &imgXX, cv::Mat &imgXY, cv::Mat &imgYY) {
    cv::multiply(grad[0], grad[0], imgXX);
    cv::multiply(grad[0], grad[1], imgXY);
    cv::multiply(grad[1], grad[1], imgYY);

#if 0
    rld::Gaussian::imgGaussianSmooth(imgXX, imgXX, rld::GAUSSIAN_KERNEL_TENSOR);
    rld::Gaussian::imgGaussianSmooth(imgXY, imgXY, rld::GAUSSIAN_KERNEL_TENSOR);
    rld::Gaussian::imgGaussianSmooth(imgYY, imgYY, rld::GAUSSIAN_KERNEL_TENSOR);
#else
    rld::Gaussian::imgAnisotropicGaussianSmoothByBirdView(imgXX, imgXX, rld::GAUSSIAN_KERNEL_TENSOR);
    rld::Gaussian::imgAnisotropicGaussianSmoothByBirdView(imgXY, imgXY, rld::GAUSSIAN_KERNEL_TENSOR);
    rld::Gaussian::imgAnisotropicGaussianSmoothByBirdView(imgYY, imgYY, rld::GAUSSIAN_KERNEL_TENSOR);
#endif
}

static void calEigenSolver(void *_imgXX, void *_imgXY, void *_imgYY, void *_grad,
                           void *_dominantVt, void *_eigenVal, int beginRow, int endRow, int cols) {
    cv::Mat *imgXX = (cv::Mat *)_imgXX;
    cv::Mat *imgXY = (cv::Mat *)_imgXY;
    cv::Mat *imgYY = (cv::Mat *)_imgYY;

    std::vector<cv::Mat> *dominantVt = (std::vector<cv::Mat> *)_dominantVt;
    std::vector<cv::Mat> *eigenVal = (std::vector<cv::Mat> *)_eigenVal;
    std::vector<cv::Mat> *grad = (std::vector<cv::Mat> *)_grad;

    for (int i = beginRow; i <= endRow; i++) {
        for (int j = 0; j < cols; j++) {
#if 0
            /* Use Eigen Solver */
            Eigen::Matrix2f tensorSt(2, 2);
            tensorSt << imgXX->at<float>(i, j), imgXY->at<float>(i, j), imgXY->at<float>(i, j), imgYY->at<float>(i, j);

            Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f> eigenSolver(tensorSt);
            Eigen::Vector2f eigenValue = eigenSolver.eigenvalues().real();
            Eigen::Matrix2f eigenVector = eigenSolver.eigenvectors().real();

            int idx = eigenValue(0) > eigenValue(1) ? 0 : 1;
            auto dominantVector = eigenVector.col(idx);
#else
            /* Use Common formula |A-eigenValue*I| = 0 */
            float c = imgXX->at<float>(i, j) * imgYY->at<float>(i, j) - imgXY->at<float>(i, j) * imgXY->at<float>(i, j);
            float b = imgXX->at<float>(i, j) + imgYY->at<float>(i, j);
            float delta = sqrt(b * b - 4 * c);
            if (delta < 1e-10) {
                delta = 0;
            }
            float x1 = (b + delta) / 2;
            float x2 = (b - delta) / 2;

            Eigen::Vector2f eigenValue(x1, x2);
            Eigen::Vector2f dominantVector(x1 - imgYY->at<float>(i, j), imgXY->at<float>(i, j));

            float tmp = sqrt(dominantVector(0) * dominantVector(0) + dominantVector(1) * dominantVector(1));
            tmp = (tmp > 1e-9) ? tmp : 1e-9;
            dominantVector /= tmp;
#endif

            float sign = dominantVector(0) * (*grad)[0].at<float>(i, j) + dominantVector(1) * (*grad)[1].at<float>(i, j);
            if (fabs(sign) < 1e-9) {
                dominantVector *= 0;
            } else if (sign < -1e-9) {
                dominantVector *= -1;
            }

            (*dominantVt)[0].at<float>(i, j) = dominantVector(0);
            (*dominantVt)[1].at<float>(i, j) = dominantVector(1);
            (*eigenVal)[0].at<float>(i, j) = eigenValue(0);
            (*eigenVal)[1].at<float>(i, j) = eigenValue(1);
        }
    }
}

#define EIGEN_SOLVER_THREADS_NUMBER 5
static std::vector<cv::Mat> computeDominantVector(cv::Mat imgXX, cv::Mat imgXY, cv::Mat imgYY, std::vector<cv::Mat> grad, std::vector<cv::Mat> &eigenVal) {
    int rows = imgXX.rows;
    int cols = imgXX.cols;
    std::vector<cv::Mat> dominantVt = {cv::Mat(rows, cols, CV_32F), cv::Mat(rows, cols, CV_32F)};
    eigenVal = {cv::Mat(rows, cols, CV_32F), cv::Mat(rows, cols, CV_32F)};

    std::thread threads[EIGEN_SOLVER_THREADS_NUMBER];
    int beginRow = 0;
    int endRow = 0;
    int amount = (int)(rows / EIGEN_SOLVER_THREADS_NUMBER);

    for (int i = 0; i < EIGEN_SOLVER_THREADS_NUMBER; i++) {
        endRow = beginRow + amount;
        if (endRow >= rows || i == EIGEN_SOLVER_THREADS_NUMBER - 1) {
            endRow = rows - 1;
        }
        threads[i] = std::thread(calEigenSolver, (void *)&imgXX, (void *)&imgXY, (void *)&imgYY, (void *)&grad,
                                 (void *)&dominantVt, (void *)&eigenVal, beginRow, endRow, cols);
        beginRow = endRow + 1;
    }

    for (int i = 0; i < EIGEN_SOLVER_THREADS_NUMBER; i++) {
        threads[i].join();
    }

    return dominantVt;
}

static cv::Mat computeDivergence(cv::Mat vectorU, cv::Mat vectorV) {
    cv::Mat gXVectorU, gYVectorV;

    cv::Mat kernX = (cv::Mat_<float>(1, 3) << -0.5, 0, 0.5);
    cv::Mat kernY = (cv::Mat_<float>(3, 1) << -0.5, 0, 0.5);
    filter2D(vectorU, gXVectorU, vectorU.depth(), kernX, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
    filter2D(vectorV, gYVectorV, vectorV.depth(), kernY, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);

    return gXVectorU + gYVectorV;
}

cv::Mat ridgeDetection(cv::Mat img, string imgPath, bool isVisualizeStepImg) {
    rld::imwrite(img, imgPath + "/00_Origin_Image.jpg", isVisualizeStepImg);
    /******************** PROCESS ********************/
    /***** Pre-process *****/
    preProcessAlgorithm(img, img, imgPath, isVisualizeStepImg);

    /***** Implement algorithm *****/
    /*** Step 1: Gaussian smoothing ***/
    // rld::Gaussian::imgGaussianSmooth(img, img, rld::GAUSSIAN_KERNEL_COMMON);
    // rld::Gaussian::imgAnisotropicGaussianSmooth(img, img, rld::GAUSSIAN_KERNEL_COMMON);
    rld::Gaussian::imgAnisotropicGaussianSmoothByBirdView(img, img, rld::GAUSSIAN_KERNEL_COMMON);
    rld::imwrite(img, imgPath + "/02_Smooth_Image.jpg", isVisualizeStepImg);

    /***  Step 2: Compute derivatives => gradient vector field ***/
    std::vector<cv::Mat> grad = computeDerivatives(img);
    rld::imwrite(grad[0], imgPath + "/03_01_Gradient_X.jpg", isVisualizeStepImg);
    rld::imwrite(grad[1], imgPath + "/03_02_Gradient_Y.jpg", isVisualizeStepImg);

    /*** Step 3: Build structure tensor ***/
    cv::Mat imgXX, imgXY, imgYY;
    computeTensorStruct(grad, imgXX, imgXY, imgYY);

    /*** Step 4: Get dominant vector ***/
    std::vector<cv::Mat> eigenValue;
    std::vector<cv::Mat> dominantVt = computeDominantVector(imgXX, imgXY, imgYY, grad, eigenValue);

    /*** Step 5: Compute divergence ***/
    cv::Mat ridgeImg = computeDivergence(dominantVt[0], dominantVt[1]) * -1;
    ridgeImg.setTo(0, ridgeImg < 0.25f);
    rld::imwrite(ridgeImg * 128, imgPath + "/04_Origin_Ridge_Image.jpg", isVisualizeStepImg);

    /*** Step 6: Apply ridge filter ***/
    /* Noise filter */
    if (IS_USING_RIDGE_NOISE_FILTER) {
        if (config.common.isVisualizeFilterImg) {
        }
        ridgeNoiseFilter(ridgeImg, ridgeImg);
        rld::imwrite(ridgeImg * 128, imgPath + "/05_02_Filter_Noise_Ridge_Image.jpg", isVisualizeStepImg);
    }

    /* Theta filter: Discard ridge point with large horizontal component */
    if (IS_USING_RIDGE_THETA_FILTER) {
        if (config.common.isVisualizeFilterImg) {
            cv::Mat visualImage = visualizeImageRidgeThetaFilter(ridgeImg, dominantVt);
            rld::imwrite(visualImage, imgPath + "/05_01_Visualize_Theta_Ridge.jpg", isVisualizeStepImg);
        }
        ridgeThetaFilter(ridgeImg, ridgeImg, dominantVt);
        rld::imwrite(ridgeImg * 128, imgPath + "/05_01_Filter_Theta_Ridge_Image.jpg", isVisualizeStepImg);
    }

    /* Confident filter */
    if (IS_USING_RIDGE_CONFIDENCE_FILTER) {
        if (config.common.isVisualizeFilterImg) {
            // cv::Mat visualImage = visualizeImageRidgeConfidenceFilter(ridgeImg, eigenValue);
            // rld::imwrite(visualImage, imgPath + "/05_03_Visualize_Confidence_Ridge.jpg", isVisualizeStepImg);
        }
        ridgeConfidenceFilter(ridgeImg, ridgeImg, eigenValue);
        rld::imwrite(ridgeImg * 128, imgPath + "/05_03_Filter_Confidence_Ridge_Image.jpg", isVisualizeStepImg);
    }
    return ridgeImg * 128;
}