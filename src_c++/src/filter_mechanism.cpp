#include "filter_mechanism.hpp"

#include <cmath>

#include "include.hpp"

extern rld::Config config;

#ifdef USE_CUDA
extern rld::Gpu gpu;
#endif

/**************************************************************************************************
 *                                                                                                *
 *                                     RIDGE THETA FILTER                                         *
 *                                                                                                *
 **************************************************************************************************/
void ridgeThetaFilter(cv::Mat srcImg, cv::Mat &dstImg, std::vector<cv::Mat> dominantVt) {
    int rows = srcImg.rows;
    int cols = srcImg.cols;
    cv::Mat angle(rows, cols, CV_32F);

    /* configuration threshold */
    std::pair<float, float> angleRange1(3.0 * M_PI / 4, 5.0 * M_PI / 4);
    // std::pair<float, float> angleRange1(0.45 * M_PI, 0.55 * M_PI);
    // std::pair<float, float> angleRange2(1.45 * M_PI, 1.55 * M_PI);

    cv::phase(dominantVt[0], dominantVt[1], angle);
    srcImg.setTo(0, ((angle >= angleRange1.first) & (angle <= angleRange1.second)));
    // srcImg.setTo(0, (((angle >= angleRange1.first) & (angle <= angleRange1.second)) | ((angle >= angleRange2.first) & (angle <= angleRange2.second)));
    srcImg.copyTo(dstImg);
}

cv::Mat visualizeImageRidgeThetaFilter(const cv::Mat srcImg, const std::vector<cv::Mat> dominantVector) {
    /* configure visualize image */
    int upsize = 4;
    cv::Mat visualizeImage(srcImg.rows * upsize, srcImg.cols * upsize, CV_8UC3);
    visualizeImage.setTo(0);

    /* configure threshold */
    std::pair<float, float> angleRange1(3.0 * M_PI / 4, 5.0 * M_PI / 4);
    // std::pair<float, float> angleRange1(0.45 * M_PI, 0.55 * M_PI);
    // std::pair<float, float> angleRange2(1.45 * M_PI, 1.55 * M_PI);

    int arrow_len = 10;
    cv::Mat angle_(srcImg.rows, srcImg.cols, CV_32F);
    cv::phase(dominantVector[0], dominantVector[1], angle_);

    for (int i = 0; i < srcImg.rows; i = i + 1)
        for (int j = 0; j < srcImg.cols; j = j + 1) {
            if (srcImg.at<float>(i, j) > 0) {
                double angle = angle_.at<float>(i, j);
                double tmp = min(fabs(angle), fabs(M_PI - angle));
                float angle_dist = (min(fabs(2 * M_PI - angle), tmp)) / M_PI * 180;

                int current_arrow_len = arrow_len;
                int current_thickness = 1;
                cv::Scalar currentcolor(255 - angle_dist, 255 - angle_dist, 255 - angle_dist);
                // if (((angle >= angleRange1.first) & (angle <= angleRange1.second)) | ((angle >= angleRange2.first) & (angle <= angleRange2.second))) {
                if ((angle >= angleRange1.first) & (angle <= angleRange1.second)) {
                    // current_arrow_len=arrow_len_2;
                    current_thickness = 1;
                    currentcolor = cv::Scalar(0, 0, 255);
                }
                cv::Point2f point2(j * upsize, i * upsize);
                cv::Point2f point1(j * upsize - cos(angle) * (current_arrow_len - angle_dist * 0.02), (i * upsize - sin(angle) * (current_arrow_len - angle_dist * 0.02)));
                cv::arrowedLine(visualizeImage, point1, point2, currentcolor, current_thickness);
            }
        }
    return visualizeImage;
}

/**************************************************************************************************
 *                                                                                                *
 *                                     RIDGE CONFIDENCE FILTER                                    *
 *                                                                                                *
 **************************************************************************************************/
template <typename T>
inline static void ridgeConfidenceFilterProcess(T srcImg, T &dstImg, const std::vector<T> eigenValue,
                                                T tmpImg, const double pow, const float c) {
    rld::subtract<T>(eigenValue[0], eigenValue[1], tmpImg);
    rld::pow<T>(tmpImg, pow, tmpImg);
    rld::divide<T>(tmpImg, c, tmpImg, -1);
    rld::exp<T>(tmpImg, tmpImg);
    rld::subtract<T>(1, tmpImg, tmpImg);
    rld::multiply<T>(srcImg, tmpImg, srcImg);

    srcImg.copyTo(dstImg);
}

void ridgeConfidenceFilter(cv::Mat srcImg, cv::Mat &dstImg, const std::vector<cv::Mat> eigenValue) {
    /* configuration threshold */
    float rangeThreshold = 0.5;
    double pow = config.confidenceFt.pow;
    float c = config.confidenceFt.c;

    cv::Mat tmp(srcImg.rows, srcImg.cols, CV_32FC1);
    ridgeConfidenceFilterProcess<cv::Mat>(srcImg, srcImg, eigenValue, tmp, pow, c);
    srcImg.setTo(0, srcImg < rangeThreshold);
    srcImg.copyTo(dstImg);
}

#ifdef USE_CUDA
void ridgeConfidenceFilter(cv::cuda::GpuMat srcImg, cv::cuda::GpuMat &dstImg, const std::vector<cv::cuda::GpuMat> eigenValue) {
    /* configuration threshold */
    float rangeThreshold = 0.5;
    double pow = config.confidenceFt.pow;
    float c = config.confidenceFt.c;

    ridgeConfidenceFilterProcess<cv::cuda::GpuMat>(srcImg, srcImg, eigenValue, gpu.tmpImg[0], pow, c);
    cv::cuda::compare(srcImg, rangeThreshold, gpu.mask, CV_CMP_LT);
    srcImg.setTo(0, gpu.mask);
    srcImg.copyTo(dstImg);
}
#endif

cv::Mat visualizeImageRidgeConfidenceFilter(const cv::Mat srcImg, const std::vector<cv::Mat> eigenValue) {
    int rows = srcImg.rows;
    int cols = srcImg.cols;

    /* configure visualize image */
    cv::Mat visualizeImage(rows, cols, CV_8UC3);
    visualizeImage.setTo(0);

    /* configure threshold */
    float rangeThreshold = 0.5;
    int pow = config.confidenceFt.pow;
    float c = config.confidenceFt.c;

    cv::Mat tmp(rows, cols, CV_32FC1);
    tmp = eigenValue[0] - eigenValue[1];
    cv::pow(tmp, pow, tmp);
    tmp = -tmp / c;
    cv::exp(tmp, tmp);
    tmp = 1 - tmp;
    cv::multiply(srcImg, tmp, tmp);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (0 < tmp.at<float>(i, j) && tmp.at<float>(i, j) < rangeThreshold) {
                visualizeImage.at<uchar>(i, j, 0) = 0;
                visualizeImage.at<uchar>(i, j, 1) = 0;
                visualizeImage.at<uchar>(i, j, 2) = 255;
            } else if (tmp.at<float>(i, j) >= rangeThreshold) {
                visualizeImage.at<uchar>(i, j, 0) = 255;
                visualizeImage.at<uchar>(i, j, 1) = 255;
                visualizeImage.at<uchar>(i, j, 2) = 255;
            }
        }
    }

    return visualizeImage;
}

/**************************************************************************************************
 *                                                                                                *
 *                                     RIDGE NOISE FILTER                                         *
 *                                                                                                *
 **************************************************************************************************/
void ridgeNoiseFilter(cv::Mat srcImg, cv::Mat &dstImg) {
    int rows = srcImg.rows;
    int cols = srcImg.cols;

    /* configuration threshold */
    int numPixelThreshold = (rows + cols) * 4;
    float valPixelThreshold;

    /* configuration histogram parameter */
    cv::Mat hist;
    float binSize = 0.1;
    int histSize = 4 / binSize;
    float range[] = {-2, 2};
    const float *histRange = {range};

    bool uniform = true;      // set bins to have the same size
    bool accumulate = false;  // clear the histograms in the beginning

    cv::calcHist(&srcImg, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);
    int currPixel = 0;
    int currBin = histSize - 1;
    while (currBin > -2) {
        currPixel += hist.at<float>(currBin);
        if (currPixel > numPixelThreshold) {
            break;
        }
        currBin--;
    }

    valPixelThreshold = (float)currBin / 10 - 2;
    srcImg.setTo(0, srcImg < valPixelThreshold);
    srcImg.copyTo(dstImg);
}