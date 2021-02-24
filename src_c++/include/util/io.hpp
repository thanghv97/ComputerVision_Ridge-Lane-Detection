#ifndef RLD_UTIL_IO_HPP
#define RLD_UTIL_IO_HPP

#include <opencv2/core/mat.hpp>
#include <opencv2/videoio.hpp>

#ifdef USE_CUDA
#include <opencv2/core/cuda.hpp>
#endif

namespace rld {

int imread(cv::Mat &img, std::string imgPath, int flag);
void imwrite(cv::Mat img, std::string imgPath, bool isWrite);
void imwriteSideBySide(cv::Mat leftImg, cv::Mat rightImg, std::string imgPath, bool isWrite);
void imshow(cv::Mat img, std::string title, bool isShow, bool isWait);
void imshowSideBySide(cv::Mat leftImg, cv::Mat rightImg, std::string title, bool isShow);
int videoCapture(cv::VideoCapture &cap, std::string vidPath);

#ifdef USE_CUDA
void imwrite(cv::cuda::GpuMat img, std::string imgPath, bool isWrite);
#endif

}  // namespace rld

#endif