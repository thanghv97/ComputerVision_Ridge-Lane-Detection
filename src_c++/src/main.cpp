#include <dirent.h>

#include <chrono>
#include <iostream>

#include "common/smooth.hpp"
#include "common/view.hpp"
#include "cuda_lane_detection.hpp"
#include "include.hpp"
#include "lane_detection.hpp"

extern rld::Config config;
extern rld::Gaussian gaussian;

#ifdef USE_CUDA
extern rld::Gpu gpu;
#endif

bool isFirstLog = false;
bool isFirstInit = false;

/**************************************************************************************************
 *                                                                                                *
 *                                              PRE_PROCESS                                       *
 *                                                                                                *
 **************************************************************************************************/
template <typename T>
void preProcess(T srcImg, T &dstImg) {
    /******************** PROCESS ********************/
    /***** Resize image *****/
#ifdef USE_CUDA
    cv::cuda::resize(srcImg, srcImg, cv::Size(), config.common.scaleResize, config.common.scaleResize, cv::INTER_AREA);
#else
    cv::resize(srcImg, srcImg, cv::Size(), config.common.scaleResize, config.common.scaleResize, cv::INTER_AREA);
#endif
    if (!isFirstLog) {
        cout << "INFO:\t\t=> Resize shape (" << srcImg.size() << ")" << endl;
    }

    /***** Crop image *****/
    cv::Rect regionOfInterest = cv::Rect((int)(srcImg.cols * config.common.rOIDyBegin),
                                         (int)(srcImg.rows * config.common.rOIDxBegin),
                                         (int)(srcImg.cols * (config.common.rOIDyEnd - config.common.rOIDyBegin)),
                                         (int)(srcImg.rows * (config.common.rOIDxEnd - config.common.rOIDxBegin)));
    srcImg = srcImg(regionOfInterest);
    if (!isFirstLog) {
        cout << "INFO:\t\t=> Crop ROI shape (" << srcImg.size() << ")" << endl;
    }

    /***** Init only first *****/
    if (!isFirstInit) {
        /* init gaussian kernel */
        gaussian.calGaussianRatio(srcImg.rows, srcImg.cols, rld::GAUSSIAN_KERNEL_COMMON, gaussian.gausRatio);
        gaussian.calGaussianRatio(srcImg.rows, srcImg.cols, rld::GAUSSIAN_KERNEL_TENSOR, gaussian.gausRatioTensor);

        /* init gpu mat if use gpu */
#ifdef USE_CUDA
        gpu.init(srcImg.rows, srcImg.cols);
#endif
        /* change state init first done is true*/
        isFirstInit = true;
    }

    isFirstLog = true;
    srcImg.copyTo(dstImg);
}

string getUrlFromUri(string uri) {
    size_t pos = uri.find_last_of("/");
    return uri.substr(0, pos);
}

/**************************************************************************************************
 *                                                                                                *
 *                                              TEST                                              *
 *                                                                                                *
 **************************************************************************************************/
/**
 *  IMAGE TO IMAGE 
 */
void testImage2Image(string inputUri, string outputUri) {
    cv::Mat img, rsImg;
    string outputUrl;
    int oriRows, oriCols;

    /******************** PROCESS ********************/
    /***** Read image *****/
    if (-1 == rld::imread(img, inputUri, cv::IMREAD_COLOR)) {
        return;
    }
    if (!isFirstLog) {
        cout << "INFO:\tImage shape (" << img.size() << ") " << endl;
    }
    oriRows = img.rows;
    oriCols = img.cols;

    /***** Get output directory *****/
    outputUrl = getUrlFromUri(outputUri);

    /***** Process *****/
    chrono::steady_clock::time_point startTimer = chrono::steady_clock::now();

#ifdef USE_CUDA
    gpu.frame.upload(img);
    /* pre-process */
    preProcess<cv::cuda::GpuMat>(gpu.frame, gpu.frame);

    /* lane detection */
    gpu.rsImg = ridgeDetection(gpu.frame, outputUrl, config.common.isVisualizeStepImg);
    gpu.rsImg.download(rsImg);
#else
    /* pre-process */
    preProcess<cv::Mat>(img, img);

    /* lane detection */
    rsImg = ridgeDetection(img, outputUrl, config.common.isVisualizeStepImg);
#endif

    chrono::steady_clock::time_point endTimer = chrono::steady_clock::now();
    chrono::duration<double> prcsTimer = chrono::duration_cast<chrono::duration<double>>(endTimer - startTimer);
    cout << "INFO:\tSolve time cost = " << prcsTimer.count() << " seconds. " << endl;

    /***** Write image *****/
    cv::Mat wImg = cv::Mat(oriRows, oriCols, CV_8U);
    wImg.setTo(0);

    cv::resize(rsImg, rsImg, cv::Size(), 1 / config.common.scaleResize, 1 / config.common.scaleResize);
    cv::Mat cropImg = wImg.colRange(oriCols / 2 - rsImg.cols / 2, oriCols / 2 - rsImg.cols / 2 + rsImg.cols).rowRange(oriRows - rsImg.rows, oriRows);
    rsImg.convertTo(rsImg, CV_8U);
    rsImg.copyTo(cropImg);
    rld::imwrite(wImg, outputUri, true);
}

/**
 *  IMAGES TO IMAGES
 */
void testImages2Images(string inputUrl, string outputUrl) {
    DIR *dir;
    struct dirent *file;

    if ((dir = opendir(inputUrl.c_str())) != NULL) {
        while ((file = readdir(dir)) != NULL) {
            string inputUri = inputUrl + file->d_name;
            string outputUri = outputUrl + file->d_name;
            testImage2Image(inputUri, outputUri);
        }
    } else {
        cout << "ERROR:\tCannot open directory ( " << inputUrl << " ) !!!" << endl;
    }
}

/**
 *  IMAGES TO VIDEO
 */
// void testImages2Video(string inputUrl, string outputUri) {
// }

/**
 *  VIDEO TO VIDEO
 */
void testVideo2Video(string inputUri, string outputUri) {
    cv::VideoCapture cap;
    cv::VideoWriter video;
    int i = 0;

    /******************** PROCESS ********************/
    /***** Read video *****/
    if (-1 == rld::videoCapture(cap, inputUri)) {
        return;
    }
    cout << "INFO:\tVideo shape ( height[" << cap.get(CV_CAP_PROP_FRAME_HEIGHT)
         << "], width[" << cap.get(CV_CAP_PROP_FRAME_WIDTH)
         << "], fps[" << cap.get(CV_CAP_PROP_FPS)
         << "] ) " << endl;

    /***** Create video writer *****/
    int height = (int)cap.get(CV_CAP_PROP_FRAME_HEIGHT) * config.common.scaleResize;
    int width = (int)cap.get(CV_CAP_PROP_FRAME_WIDTH) * config.common.scaleResize;
    video = cv::VideoWriter(outputUri, CV_FOURCC('M', 'J', 'P', 'G'), cap.get(CV_CAP_PROP_FPS),
                            cv::Size(width * 2, int(height / 2) + 130));

    /***** Get Get output directory *****/
    string outputUrl = getUrlFromUri(outputUri);

    /***** Process *****/
    chrono::steady_clock::time_point startTimer = chrono::steady_clock::now();

    while (1) {
        cv::Mat frame, rsImg, combined;

        /* read frame */
        cap >> frame;
        if (frame.empty()) {
            break;
        }
        // cv::rotate(frame, frame, ROTATE_90_COUNTERCLOCKWISE);

        /* create write frame */
        cv::Mat writeFrame(int(height / 2) + 130, width * 2, CV_8UC3);
        writeFrame.setTo(0);

#ifdef USE_CUDA
        gpu.frame.upload(frame);
        /* pre-process */
        preProcess<cv::cuda::GpuMat>(gpu.frame, gpu.frame);
        gpu.frame.download(frame);

        /* lane detection */
        gpu.rsImg = ridgeDetection(gpu.frame, outputUrl, config.common.isVisualizeStepImg);
        gpu.rsImg.download(rsImg);
#else
        /* pre-process */
        preProcess<cv::Mat>(frame, frame);

        /* lane detection */
        rsImg = ridgeDetection(frame, outputUrl, config.common.isVisualizeStepImg);
#endif

        if (i % 100 == 0) {
            cout << "INFO:\tframe " << i << "..." << endl;
        }
        i++;

        /* put ridge detection result */
        rsImg.convertTo(rsImg, CV_8U);
        cv::cvtColor(rsImg, rsImg, cv::COLOR_GRAY2BGR);
        cv::hconcat(frame, rsImg, combined);
        cv::Mat pt = writeFrame.colRange(0, combined.cols).rowRange(0, combined.rows);
        combined.copyTo(pt);

        /* put parameter turned */
        int putText_row = int(height / 2);
        putText(writeFrame, "Gaussian Smooth: type=anisotropic, kernel_size=" + to_string(gaussian.gausKCommon.kSize) + ", sigmaX=" + to_string(gaussian.gausKCommon.sigmaX) + ", sigmaY=" + to_string(gaussian.gausKCommon.sigmaY),
                cv::Point(5, putText_row + 30), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(255, 255, 255), 0.5);
        putText(writeFrame, "Gaussian Tensor: type=anisotropic, kernel_size=" + to_string(gaussian.gausKTensor.kSize) + ", sigmaX=" + to_string(gaussian.gausKTensor.sigmaX) + ", sigmaY=" + to_string(gaussian.gausKTensor.sigmaY),
                cv::Point(5, putText_row + 55), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(255, 255, 255), 0.5);
        putText(writeFrame, "Scale: " + to_string(config.common.scaleResize),
                cv::Point(5, putText_row + 80), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(255, 255, 255), 0.5);
        putText(writeFrame, "Confident: powSize=" + to_string(config.confidenceFt.pow) + ", C=" + to_string(config.confidenceFt.c) + "",
                cv::Point(5, putText_row + 105), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(255, 255, 255), 0.5);
        putText(writeFrame, "Theta Filter: no",
                cv::Point(5, putText_row + 130), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(255, 255, 255), 0.5);

        if (i == 1) {
            rld::imshow(writeFrame, "writeFrame", true, false);
        }
        video.write(writeFrame);
    }

    chrono::steady_clock::time_point endTimer = chrono::steady_clock::now();
    chrono::duration<double> prcsTimer = chrono::duration_cast<chrono::duration<double>>(endTimer - startTimer);
    cout << "INFO:\tSolve time cost = " << prcsTimer.count() << " seconds. " << endl;

    /***** Release *****/
    cap.release();
    video.release();
}

/**
 * VIDEO TO IMAGES
 */
void testVideo2Images(string inputUri, string outputUrl, int framePerImg) {
    cv::VideoCapture cap;
    int i = 0;

    /******************** PROCESS ********************/
    /***** Read video *****/
    if (-1 == rld::videoCapture(cap, inputUri)) {
        return;
    }
    cout << "INFO:\tVideo shape ( height[" << cap.get(CV_CAP_PROP_FRAME_HEIGHT)
         << "], width[" << cap.get(CV_CAP_PROP_FRAME_WIDTH)
         << "], fps[" << cap.get(CV_CAP_PROP_FPS)
         << "] ) " << endl;

    /***** Process *****/
    chrono::steady_clock::time_point startTimer = chrono::steady_clock::now();

    while (1) {
        cv::Mat frame, rsImg, combined;

        /* read frame */
        cap >> frame;
        if (frame.empty()) {
            break;
        }
        // cv::rotate(frame, frame, ROTATE_90_COUNTERCLOCKWISE);

        /* write image per framePerImg */
        if (i % framePerImg == 0) {
            string outputUri = outputUrl + to_string(i / framePerImg) + ".jpg";

            rld::imwrite(frame, outputUri, true);
            testImage2Image(outputUri, outputUri);
        }
        i++;
    }

    chrono::steady_clock::time_point endTimer = chrono::steady_clock::now();
    chrono::duration<double> prcsTimer = chrono::duration_cast<chrono::duration<double>>(endTimer - startTimer);
    cout << "INFO:\tSolve time cost = " << prcsTimer.count() << " seconds. " << endl;

    /***** Release *****/
    cap.release();
}

int main(int argc, char *argv[]) {
    int inputOpt = -1;
    string cfgUrl = "";
    string input = "";
    string output = "";

    /******************** GET PARAMETER ********************/
    if (argc < 5) {
        cout << "ERROR:\tMissing argument!!!" << endl;
        cout << "ERROR:\t\t[bin] [OPT] [CFG_URL] [InputUri] [OutputUrl]" << endl;
        return -1;
    }

    cfgUrl = argv[1];
    inputOpt = atoi(argv[2]);
    input = argv[3];
    output = argv[4];

#ifdef USE_CUDA
    /***** Check CUDA Device *****/
    int deviceGpu = cv::cuda::getCudaEnabledDeviceCount();
    if (deviceGpu > 0) {
        /* show CUDA device information */
        deviceGpu = cv::cuda::getDevice();
        cv::cuda::printCudaDeviceInfo(deviceGpu);
    } else {
        cout << "INFO:\tNot found cuda device - Check nvidia driver or cmake don't use CUDA" << endl;
        return -1;
    }
#endif

    /***** Get config turning parameter *****/
    rld::getConfig(cfgUrl);

    /***** Test *****/
    if (inputOpt == 0) {
        testImage2Image(input, output);
    } else if (inputOpt == 1) {
        testImages2Images(input, output);
    } else if (inputOpt == 3) {
        int framePerImg = atoi(argv[5]);
        testVideo2Images(input, output, framePerImg);
    } else if (inputOpt == 4) {
        testVideo2Video(input, output);
    } else {
        cout << "ERROR:\tMissing argument!!! - input option [ 0 for image, 1 for video ]" << endl;
    }

    return 0;
}