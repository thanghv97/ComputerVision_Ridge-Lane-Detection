#include "shadow_removal.hpp"

#include <cmath>
#include <iomanip>

#include "include.hpp"
#include "normalize.hpp"

rld::ShadowRemoval::ShadowRemoval() {
}

rld::ShadowRemoval::~ShadowRemoval() {
}

static std::vector<cv::Mat> computeLogChromaticity(cv::Mat img) {
    int rows = img.rows;
    int cols = img.cols;

    std::vector<cv::Mat> xPlane = {cv::Mat(rows, cols, CV_32F), cv::Mat(rows, cols, CV_32F)};
    std::vector<cv::Mat> channel;
    cv::split(img, channel);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            channel[0].at<uchar>(i, j) = (channel[0].at<uchar>(i, j) == 0) ? 1 : channel[0].at<uchar>(i, j);
            channel[1].at<uchar>(i, j) = (channel[1].at<uchar>(i, j) == 0) ? 1 : channel[1].at<uchar>(i, j);
            channel[2].at<uchar>(i, j) = (channel[2].at<uchar>(i, j) == 0) ? 1 : channel[2].at<uchar>(i, j);

            float geoMean = cbrt(channel[0].at<uchar>(i, j) * channel[1].at<uchar>(i, j) * channel[2].at<uchar>(i, j));

            float logR = log(((float)channel[0].at<uchar>(i, j) / geoMean));
            float logG = log(((float)channel[1].at<uchar>(i, j) / geoMean));
            float logB = log(((float)channel[2].at<uchar>(i, j) / geoMean));

            /* orthogonal matrix 
                [[1/sqrt(2), -1/sqrt(2), 0         ]
                [1/sqrt(6),  1/sqrt(6), -2/sqrt(6)]]
            */
            xPlane[0].at<float>(i, j) = (1 / sqrt(2)) * logR - (1 / sqrt(2)) * logG;
            xPlane[1].at<float>(i, j) = (1 / sqrt(6)) * logR + (1 / sqrt(6)) * logG - (2 / sqrt(6)) * logB;
        }
    }

    return xPlane;
}

inline static cv::Mat computeGrayImg(const std::vector<cv::Mat> xPlane, const int alpha) {
    int rows = xPlane[0].rows;
    int cols = xPlane[0].cols;
    cv::Mat grayImg(rows, cols, CV_32F);

    float angle = M_PI * alpha / 180;
    grayImg = xPlane[0] * cos(angle) + xPlane[1] * sin(angle);

    return grayImg;
}

inline static void getMiddleNonOuliersGrayImg(cv::Mat img, float &mean, float &stdv, float &lowerBound, float &upperBound) {
    // reference Chebyshev's Theorem
    // get parameter for compute confidence interval [mean-k*std, mean+k*std]
    float k = 3;  // for 90 %
    cv::Scalar means, stdvs;

    img = img.reshape(1, img.rows * img.cols);

    /* Get non-oulier gray img */
    cv::meanStdDev(img, means, stdvs);

    lowerBound = means[0] - k * stdvs[0];
    upperBound = means[0] + k * stdvs[0];

    /* Get 90% middle non-outlier gray img */
    img.setTo(lowerBound - 1, ((img < lowerBound) | (img > upperBound)));
    cv::sort(img, img, CV_SORT_EVERY_COLUMN + CV_SORT_ASCENDING);
    int lowerIdx = 0;
    int upperIdx = img.rows * img.cols;

    for (int i = 0; i < img.rows * img.cols; i++) {
        if (img.at<float>(i, 0) != lowerBound - 1) {
            break;
        }
        lowerIdx = i;
    }

    int dist = upperIdx - lowerIdx;
    lowerIdx += (int)(dist * 5 / 100);
    upperIdx -= (int)(dist * 5 / 100);
    cv::Mat midNonOulierImg = img.rowRange(lowerIdx, upperIdx + 1);
    cv::meanStdDev(midNonOulierImg, means, stdvs);
    mean = means[0];
    stdv = stdvs[0];
    lowerBound = img.at<float>(lowerIdx, 1);
    upperBound = img.at<float>(upperIdx, 1);
}

inline static cv::Mat computeHistogram(cv::Mat img, const float lowerBound, const float upperBound, const float stdv, int &histSize) {
    int rows = img.rows;
    int cols = img.cols;
    float binWidth, range[] = {lowerBound, upperBound};
    const float *histRange = {range};
    bool uniform = true;      // set bins to have the same size
    bool accumulate = false;  // clear the histograms in the beginning
    cv::Mat hist;

    binWidth = 3.5 * stdv * pow(rows * cols, -1.0 / 3);
    histSize = floor((upperBound - lowerBound) / binWidth);  // floor for smaller binWidth, ceil for higher binWidth
    // std::cout << "=== histSize: " << histSize << std::endl;

    cv::calcHist(&img, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);

    return hist;
}

inline static float computeEntropy(const cv::Mat hist, const int histSize) {
    float totalHist, entropy = 0;
    cv::Mat p;

    totalHist = cv::sum(hist)[0];
    // std::cout << "=== totalHist: " << totalHist << std::endl;
    p = hist / totalHist;
    for (int i = 0; i < histSize; i++) {
        entropy = (hist.at<float>(i) > 0) ? entropy + p.at<float>(i) * log(p.at<float>(i)) : entropy;
    }
    entropy *= -1;

    return entropy;
}

std::pair<int, std::vector<cv::Mat>> rld::ShadowRemoval::derive1DShadowFreeImage(cv::Mat srcImg, cv::Mat &dstImg) {
    cv::Mat grayImg;
    float minEntropy = std::numeric_limits<float>::max();
    int theta = 0;

    /******************** PROCESS ********************/
    /***** Compute log-chromaticity *****/
    std::vector<cv::Mat> xPlane = computeLogChromaticity(srcImg);

    for (int i = 0; i < 180; i++) {
        /***** step 1: Obtain a grayscale image projecting log-chromaticity pixel values of alpha *****/
        grayImg = computeGrayImg(xPlane, i);
        // rld::imshow(grayImg, "result", true);

        /***** step 2: reject the ouliers in gray image according to Chebyshev’s theorem *****/
        float lowerBound, upperBound, mean, stdv;
        getMiddleNonOuliersGrayImg(grayImg, mean, stdv, lowerBound, upperBound);
        // std::cout << "=== " << i << " lowerBound: " << lowerBound << std::endl;
        // std::cout << "=== " << i << " upperBound: " << upperBound << std::endl;

        // /***** step 3: plot image gray to histogram *****/
        int histSize;
        cv::Mat hist = computeHistogram(grayImg, lowerBound, upperBound, stdv, histSize);

        // /***** step 4: Compute the entropy *****/
        float entropy = computeEntropy(hist, histSize);
        // std::cout << "=== " << i << " entropy: " << entropy << std::endl;
        if (entropy < minEntropy) {
            minEntropy = entropy;
            theta = i;
        }
    }
    theta = 140;
    dstImg = computeGrayImg(xPlane, theta);

    rld::Normalize::imgValueNormalize(dstImg, dstImg, 0, 255);

    double minVal, maxVal;
    rld::imgGetMinMaxVal(dstImg, minVal, maxVal);
    std::cout << "minVal " << minVal << " - maxVal " << maxVal << std::endl;

    std::cout << "=== theta: " << theta << std::endl;
    std::cout << "=== min entropy: " << minEntropy << std::endl;
    return std::pair<int, std::vector<cv::Mat>>(theta, xPlane);
}

static std::vector<float> computeLightningExtra(cv::Mat srcImg, const std::vector<cv::Mat> xPlane, const std::vector<cv::Mat> xPlaneTheta) {
    int rows = srcImg.rows;
    int cols = srcImg.cols;

    cv::Mat grayImg, arrImg;
    std::vector<float> extra;

    /* find min of 1% brightest pixels */
    cv::cvtColor(srcImg, grayImg, cv::COLOR_RGB2GRAY);
    arrImg = grayImg.reshape(1, rows * cols);
    cv::sort(arrImg, arrImg, CV_SORT_EVERY_COLUMN + CV_SORT_ASCENDING);
    for (int i = 0; i < rows * cols; i++) {
        std::cout << arrImg.at<float>(0, i) << std::endl;
    }

    return {0, 0};
}

void rld::ShadowRemoval::derive2DShadowFreeImage(cv::Mat srcImg, cv::Mat &dstImg, const std::vector<cv::Mat> xPlane, const int theta) {
    int rows = srcImg.rows;
    int cols = srcImg.cols;
    float angle = M_PI * theta / 180;
    std::vector<cv::Mat> xPlaneTheta = {cv::Mat(rows, cols, CV_32F), cv::Mat(rows, cols, CV_32F)};
    std::vector<cv::Mat> chroEst = {cv::Mat(rows, cols, CV_32F), cv::Mat(rows, cols, CV_32F), cv::Mat(rows, cols, CV_32F)};
    cv::Mat chroTotal(rows, cols, CV_32F);
    std::vector<float> extra;

    /* projector matrix 2x2 */
    cv::Mat prjMat = (cv::Mat_<float>(2, 2) << cos(angle) * cos(angle), cos(angle) * sin(angle), sin(angle) * cos(angle), sin(angle) * sin(angle));

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            xPlaneTheta[0].at<float>(i, j) = xPlane[0].at<float>(i, j) * prjMat.at<float>(0, 0) + xPlane[1].at<float>(i, j) * prjMat.at<float>(1, 0);
            xPlaneTheta[1].at<float>(i, j) = xPlane[0].at<float>(i, j) * prjMat.at<float>(0, 1) + xPlane[1].at<float>(i, j) * prjMat.at<float>(1, 1);
        }
    }
    
    /* compute lightning extra */
    extra = computeLightningExtra(srcImg, xPlane, xPlaneTheta);
    xPlaneTheta[0] += extra[0];
    xPlaneTheta[1] += extra[1];

#if 0
    /* orthogonal matrix
        [[1/sqrt(2), -1/sqrt(2), 0         ]
        [1/sqrt(6),  1/sqrt(6), -2/sqrt(6)]]
    */
    chroEst[0] = xPlaneTheta[0] * 1 / sqrt(2) + xPlaneTheta[1] * 1 / sqrt(6);
    chroEst[1] = -xPlaneTheta[0] * 1 / sqrt(2) + xPlaneTheta[1] * 1 / sqrt(6);
    chroEst[2] = -xPlaneTheta[1] * 2 / sqrt(6);
    chroTotal = chroEst[0] + chroEst[1] + chroEst[2];
    // for (int i = 0; i < rows; i++) {
    //     for (int j = 0; j < cols; j++) {
    //         dstImg.at<float>(i, j, 0) = chroEst[0].at<float>(i, j) / chroTotal.at<float>(i, j);
    //         dstImg.at<float>(i, j, 1) = chroEst[1].at<float>(i, j) / chroTotal.at<float>(i, j);
    //         dstImg.at<float>(i, j, 2) = chroEst[2].at<float>(i, j) / chroTotal.at<float>(i, j);
    //     }
    // }
#endif
}

void rld::ShadowRemoval::imgIlluminantInvariance(cv::Mat srcImg, cv::Mat &dstImg) {
    std::pair<int, std::vector<cv::Mat>> grayProps;
    cv::Mat _1DImg, _2DImg;

    /******************** PROCESS ********************/
    /**** 1-D shadow free image *****/
    grayProps = derive1DShadowFreeImage(srcImg, _1DImg);

    /**** 2-D shadow free image *****/
    derive2DShadowFreeImage(srcImg, _2DImg, grayProps.second, grayProps.first);

    _2DImg.copyTo(dstImg);
    rld::imshow(dstImg, "result", true, true);
}