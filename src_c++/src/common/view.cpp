#include "view.hpp"

#include "include.hpp"

extern rld::Camera camera;

rld::View view = rld::View();

rld::View::View() {
    this->bvCfg.minRow = 0;
    this->bvCfg.maxRow = 0;
    this->bvCfg.minCol = 0;
    this->bvCfg.maxCol = 0;
}

rld::View::~View() {
}

void rld::View::calBirdViewMask() {
    double alpha = camera.pitchAngle * M_PI / 180;
    double dist = camera.height / sin(alpha);

    // Projection matrix 2D -> 3D
    cv::Mat A1 = (cv::Mat_<float>(4, 3) << 1, 0, this->bvCfg.minRow,
                  0, 0, 0,
                  0, 1, this->bvCfg.minCol,
                  0, 0, 1);

    // Rotation matrices Rx, Ry, Rz
    cv::Mat RX = (cv::Mat_<float>(4, 4) << 1, 0, 0, 0,
                  0, cos(alpha), -sin(alpha), 0,
                  0, sin(alpha), cos(alpha), 0,
                  0, 0, 0, 1);
    // R - rotation matrix
    cv::Mat R = RX;
    cv::Mat T = (cv::Mat_<float>(4, 4) << 1, 0, 0, 0,
                 0, 1, 0, 0,
                 0, 0, 1, -dist,
                 0, 0, 0, 1);

    // K - intrinsic matrix
    cv::Mat K = (cv::Mat_<float>(3, 4) << camera.focalLength, this->bvCfg.skew, this->bvCfg.u0, 0,
                 0, camera.focalLength, this->bvCfg.v0, 0,
                 0, 0, 1, 0);

    this->birdViewMask = K * T * R * A1;
    this->birdViewMaskInv = this->birdViewMask.inv();
}

void rld::View::cvtCameraView2BirdView(cv::Mat srcImg, cv::Mat &dstImg) {
    if (view.bvCfg.maxCol > 0 && view.bvCfg.maxRow > 0) {
        cv::warpPerspective(srcImg, dstImg, view.birdViewMaskInv, cv::Size((int)view.bvCfg.maxCol, (int)view.bvCfg.maxRow), cv::INTER_AREA);
    } else {
        int rows = srcImg.rows;
        int cols = srcImg.cols;
        cv::warpPerspective(srcImg, dstImg, view.birdViewMaskInv, cv::Size(cols, rows * 2), cv::INTER_AREA);
    }
    cv::flip(dstImg, dstImg, 1);
}

void rld::View::cvtBirdView2CameraView(cv::Mat srcImg, cv::Mat &dstImg) {
    int rows = srcImg.rows;
    int cols = srcImg.cols;

    cv::warpPerspective(srcImg, dstImg, view.birdViewMask, cv::Size(cols, (int)(rows / 2)), cv::INTER_AREA);
}

cv::Mat rld::View::getBirdViewMask() {
    return this->birdViewMask;
}

cv::Mat rld::View::getBirdViewMaskInv() {
    return this->birdViewMaskInv;
}