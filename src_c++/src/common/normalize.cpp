#include "normalize.hpp"

#include <cmath>

#include "include.hpp"

rld::Normalize::Normalize() {
}

rld::Normalize::~Normalize() {
}

void rld::Normalize::imgValueNormalize(cv::Mat srcImg, cv::Mat &dstImg, int minValNor, int maxValNor) {
    double minVal, maxVal;

    rld::imgGetMinMaxVal(srcImg, minVal, maxVal);
    double scale = (maxValNor - minValNor) / (maxVal - minVal);
    dstImg = (srcImg - minVal) * scale + minValNor;
}

void rld::Normalize::imgHistogramNormalize(cv::Mat srcImg, cv::Mat &dstImg) {
    int histSize = 256;
    int totalPixel = srcImg.rows * srcImg.cols;
    float range[] = {0, 256};
    const float *histRange = {range};

    bool uniform = true;      // set bins to have the same size
    bool accumulate = false;  // clear the histograms in the beginning

    cv::Mat hist, p;
    cv::calcHist(&srcImg, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);
    p = hist / totalPixel;

    float t[256];
    t[0] = p.at<float>(0);
    for (int i = 1; i < histSize; i++) {
        t[i] = t[i - 1] + p.at<float>(i);
    }

    for (int i = 0; i < srcImg.rows; i++) {
        for (int j = 0; j < srcImg.cols; j++) {
            dstImg.at<uchar>(i, j) = floor(255 * t[srcImg.at<uchar>(i, j)]);
        }
    }
}