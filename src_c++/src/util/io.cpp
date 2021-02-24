#include "io.hpp"

#include <iostream>
#include <opencv2/highgui.hpp>    // for nameWindow, imshow, waitkey
#include <opencv2/imgcodecs.hpp>  // for imread, imwrite

#include "log.hpp"

int rld::imread(cv::Mat &img, std::string imgPath, int flag) {
    if (imgPath == "") {
        std::cout << "ERROR:\tInput image path empty!!!" << std::endl;
        return -1;
    }

    std::cout << "INFO:\tRead image ( " << imgPath << " )" << std::endl;
    img = cv::imread(imgPath, flag);
    if (!img.data) {
        std::cout << "ERROR:\tCannot read image!!!" << std::endl;
        return -1;
    }

    return 0;
}

void rld::imwrite(cv::Mat img, std::string imgPath, bool isWrite) {
    if (!isWrite) {
        return;
    }

    if (imgPath == "") {
        std::cout << "ERROR:\tOutput image path empty!!!" << std::endl;
        return;
    }

    if (false == cv::imwrite(imgPath, img)) {
        std::cout << "ERROR:\tFailed to write image!!! - img_path( " << imgPath << " )" << std::endl;
    }
}

void rld::imwriteSideBySide(cv::Mat leftImg, cv::Mat rightImg, std::string imgPath, bool isWrite) {
    if (!isWrite) {
        return;
    }

    if (imgPath == "") {
        std::cout << "ERROR:\tOutput image path empty!!!" << std::endl;
        return;
    }

    if ((leftImg.rows != rightImg.rows) || (leftImg.cols != rightImg.cols)) {
        std::cout << "ERROR:\tCannot show image - two images are different dimension" << std::endl;
        return;
    }

    cv::Mat combined(leftImg.rows, leftImg.cols * 2, CV_8UC3);
    combined.setTo(0);

    if (false == cv::imwrite(imgPath, combined)) {
        std::cout << "ERROR:\tFailed to write image!!! - img_path( " << imgPath << " )" << std::endl;
    }
}

void rld::imshow(cv::Mat img, std::string title, bool isShow, bool isWait) {
    if (!isShow) {
        return;
    }

    if (!img.data) {
        std::cout << "ERROR:\tCannot show image - image no data" << std::endl;
        return;
    }

    cv::namedWindow(title, cv::WINDOW_FULLSCREEN);
    cv::imshow(title, img);
    if (isWait) {
        cv::waitKey(0);
    }
}

void rld::imshowSideBySide(cv::Mat leftImg, cv::Mat rightImg, std::string title, bool isShow) {
    if (!isShow) {
        return;
    }

    if (!leftImg.data || !rightImg.data) {
        std::cout << "ERROR:\tCannot show image - image no data" << std::endl;
        return;
    }

    if ((leftImg.rows != rightImg.rows) || (leftImg.cols != rightImg.cols)) {
        std::cout << "ERROR:\tCannot show image - two images are different dimension" << std::endl;
        return;
    }

    cv::namedWindow(title, cv::WINDOW_FULLSCREEN);
    cv::Mat combined(leftImg.rows, leftImg.cols * 2, CV_8UC3);
    cv::hconcat(leftImg, rightImg, combined);
    cv::imshow(title, combined);
    cv::waitKey(0);
}

int rld::videoCapture(cv::VideoCapture &cap, std::string vidPath) {
    if (vidPath == "") {
        std::cout << "ERROR:\tInput video path empty!!!" << std::endl;
        return -1;
    }

    std::cout << "INFO:\tRead video ( " << vidPath << " )" << std::endl;
    cap = cv::VideoCapture(vidPath);
    if (!cap.isOpened()) {
        std::cout << "ERROR:\tFailed to open video stream or file!!!" << std::endl;
        return -1;
    }

    return 0;
}

#ifdef USE_CUDA
void rld::imwrite(cv::cuda::GpuMat img, std::string imgPath, bool isWrite) {
    cv::Mat tmpImg;
    img.download(tmpImg);
    rld::imwrite(tmpImg, imgPath, isWrite);
}
#endif