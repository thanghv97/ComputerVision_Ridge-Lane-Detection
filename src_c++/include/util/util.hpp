#ifndef RLD_UTIL_HPP
#define RLD_UTIL_HPP

#include <opencv2/core.hpp>      //  cv::pow
#include <opencv2/core/mat.hpp>  //  cv::Mat

#ifdef USE_CUDA
#include <opencv2/cudaarithm.hpp>  //  cv::cuda::pow
#endif

namespace rld {

/********** template **********/
/***** subtract: Img - Img *****/
template <typename T>
void subtract(T srcImg1, T srcImg2, T &dstImg) {
}

template <>
inline void subtract(cv::Mat srcImg1, cv::Mat srcImg2, cv::Mat &dstImg) {
    dstImg = srcImg1 - srcImg2;
}

#ifdef USE_CUDA
template <>
inline void subtract(cv::cuda::GpuMat srcImg1, cv::cuda::GpuMat srcImg2, cv::cuda::GpuMat &dstImg) {
    cv::cuda::subtract(srcImg1, srcImg2, dstImg);
}
#endif

/***** subtract: Number - Img *****/
template <typename T>
void subtract(double number, T srcImg, T &dstImg) {}

template <>
inline void subtract(double number, cv::Mat srcImg, cv::Mat &dstImg) {
    dstImg = number - srcImg;
}

#ifdef USE_CUDA
template <>
inline void subtract(double number, cv::cuda::GpuMat srcImg, cv::cuda::GpuMat &dstImg) {
    cv::cuda::subtract(number, srcImg, dstImg);
}
#endif

/***** multiply: Img - Img *****/
template <typename T>
void multiply(T srcImg1, T srcImg2, T &dstImg, double scale = 1) {}

template <>
inline void multiply(cv::Mat srcImg1, cv::Mat srcImg2, cv::Mat &dstImg, double scale) {
    cv::multiply(srcImg1, srcImg2, dstImg, scale);
}

#ifdef USE_CUDA
template <>
inline void multiply(cv::cuda::GpuMat srcImg1, cv::cuda::GpuMat srcImg2, cv::cuda::GpuMat &dstImg, double scale) {
    cv::cuda::multiply(srcImg1, srcImg2, dstImg, scale);
}
#endif

/***** multiply: Img - Number *****/
template <typename T>
void divide(T srcImg, double number, T &dstImg, double scale = 1) {}

template <>
inline void divide(cv::Mat srcImg, double number, cv::Mat &dstImg, double scale) {
    dstImg = (srcImg * scale) / number;
}

#ifdef USE_CUDA
template <>
inline void divide(cv::cuda::GpuMat srcImg, double number, cv::cuda::GpuMat &dstImg, double scale) {
    cv::cuda::divide(srcImg, number, dstImg, scale);
}
#endif

/***** pow: Img - Number *****/
template <typename T>
void pow(T srcImg, double power, T &dstImg) {}

template <>
inline void pow(cv::Mat srcImg, double power, cv::Mat &dstImg) {
    cv::pow(srcImg, power, dstImg);
}

#ifdef USE_CUDA
template <>
inline void pow(cv::cuda::GpuMat srcImg, double power, cv::cuda::GpuMat &dstImg) {
    cv::cuda::pow(srcImg, power, dstImg);
}
#endif

/***** exp: Img - Img *****/
template <typename T>
void exp(T srcImg, T &dstImg) {}

template <>
inline void exp(cv::Mat srcImg, cv::Mat &dstImg) {
    cv::exp(srcImg, dstImg);
}

#ifdef USE_CUDA
template <>
inline void exp(cv::cuda::GpuMat srcImg, cv::cuda::GpuMat &dstImg) {
    cv::cuda::exp(srcImg, dstImg);
}
#endif

/***** util function *****/
void imgGetMinMaxVal(cv::Mat img, double &minVal, double &maxVal);

}  // namespace rld

#endif