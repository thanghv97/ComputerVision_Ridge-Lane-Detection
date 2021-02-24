#include <dirent.h>

#include <chrono>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <string>

#include "opencv2/core/mat.hpp"         // Mat
#include "opencv2/imgproc.hpp"          // cvtColor, resize, gaussianBlur
#include "opencv2/videoio/videoio_c.h"  // CV_CAP_PROP_FRAME_WIDTH, CV_CAP_PROP_FRAME_HEIGHT, CV_CAP_PROP_FPS

using namespace std;
using namespace cv;
int testOpt = -1;
string inputUri = "";
string outputUrl = "";
/*
MKZ short camera
double alpha =1*M_PI/180.0;
double beta=-0.0775;
int focal_length=513;
int dist=874;
int u0= 287;
int v0= 556;
int skew=4548;
*/
double alpha = 1 * M_PI / 180.0;
int alpha_degree = 2;
double beta = -0.0775;
int focal_length = 1026;
int dist = 690;
int u0 = 538;
int v0 = 396;
int skew = 0;

int vAPImread(cv::Mat& img, std::string imgPath, int flag) {
    if (imgPath == "") {
        std::cout << "ERROR:\tInput image path empty!!!" << std::endl;
        return -1;
    }

    // std::cout<<"INFO:\tRead image ( "<<imgPath<<" )"<<std::endl;
    img = cv::imread(imgPath, flag);
    if (!img.data) {
        std::cout << "ERROR:\tCannot read image!!!" << std::endl;
        return -1;
    }

    return 0;
}

void convertImage2BirdView(cv::Mat& ridge_img, float alpha_, int f_, int dist_, double u_0, double v_0, double skew) {
    double focalLength, dist, alpha, beta, gamma;
    alpha = ((double)alpha_) * M_PI / 180;
    // beta =((double)beta_ ) * M_PI/180;
    // gamma =((double)gamma_ ) * M_PI/180;

    focalLength = (double)f_;
    dist = (double)dist_;

    // Projecion matrix 2D -> 3D
    Mat A1 = (Mat_<float>(4, 3) << 1, 0, 0,
              0, 0, 0,
              0, 1, 0,
              0, 0, 1);

    // Rotation matrices Rx, Ry, Rz
    Mat RX = (Mat_<float>(4, 4) << 1, 0, 0, 0,
              0, cos(alpha), -sin(alpha), 0,
              0, sin(alpha), cos(alpha), 0,
              0, 0, 0, 1);
    Mat RY = (Mat_<float>(4, 4) << cos(beta), 0, -sin(beta), 0,
              0, 1, 0, 0,
              sin(beta), 0, cos(beta), 0,
              0, 0, 0, 1);
    // R - rotation matrix
    Mat R = RX;
    Mat T = (Mat_<float>(4, 4) << 1, 0, 0, 0,
             0, 1, 0, 0,
             0, 0, 1, -dist,
             0, 0, 0, 1);

    // K - intrinsic matrix
    Mat K = (Mat_<float>(3, 4) << focalLength, skew, u_0, 0,
             0, focalLength, v_0, 0,
             0, 0, 1, 0);

    Mat transformation = K * T * R * A1;
    Mat transformation_inv = transformation.inv();

#if 0
    std::vector<cv::Point3f> list_point;
    
    for (int i=0;i<ridge_img.rows;i++)
        for (int j=0;j<ridge_img.cols;j++)
        if (ridge_img.at<uchar>(i,j)>0){
            cv::Mat point = (cv::Mat_<float>(3, 1) << j, i, 1);
            cv::Mat transform_point = transformation_inv*point;
            transform_point = transform_point / transform_point.at<float>(2, 0);
            cv::Point3f point_(transform_point.at<float>(0,0),transform_point.at<float>(1,0),ridge_img.at<uchar>(i,j));
            // cout<<point_<<endl;
            list_point.push_back(point_);

        }
    double min_row=list_point[0].y;
    double min_col=list_point[0].x;
    double max_row=list_point[0].y;
    double max_col=list_point[0].x;
    
    for (int i=1;i<list_point.size();i++)
    {
        if (min_row>list_point[i].y) min_row=list_point[i].y;
        if (max_row<list_point[i].y) max_row=list_point[i].y;
        
        if (min_col>list_point[i].x) min_col=list_point[i].x;
        if (max_col<list_point[i].x) max_col=list_point[i].x;
    }

    for (int i=0;i<list_point.size();i++)
    {
        list_point[i].x=list_point[i].x-min_col;
        list_point[i].y=list_point[i].y-min_row;
    }
    max_col=max_col-min_col;
    max_row=max_row-min_row;
    cout<<"max row col "<<max_row<<" "<<max_col<<" "<<min_col<<" "<<min_row<<endl;
    cv::Mat img(int(max_row),int(max_col),CV_8U);
    img.setTo(0);
    for (int i=0;i<list_point.size();i++){
        img.at<uchar>(int(list_point[i].y),int(list_point[i].x))=int(list_point[i].z);
    }
    cv::flip(img,img,1);
    cv::waitKey(0);

#endif

    cv::warpPerspective(ridge_img, ridge_img, transformation.inv(), cv::Size(ridge_img.cols, ridge_img.rows * 2), INTER_AREA);
    cv::flip(ridge_img, ridge_img, 1);
    cv::imwrite("/home/thanghv7/Autopilot/Source/ridge-lane-detection/tools/get_camera_params/out.jpg", ridge_img);
    // cv::imwrite("/home/thanghv7/Autopilot/Source/ridge-lane-detection/tools/get_camera_params/out2.jpg",img);
}

static void on_trackbar(int, void* data) {
    cout << "on trackbar alpha " << (alpha_degree - 90) / 2.0f << " focal length " << focal_length << " dist " << dist << " u0 " << u0 << " v0 " << v0 - 400 << " skew " << skew << endl;
    cv::Mat* img = (cv::Mat*)data;
    cv::Mat tmp = (*img).clone();
    // cv::imshow("tmp",tmp);
    // cv::waitKey(0);
    convertImage2BirdView(tmp, (alpha_degree - 90) / 2.0f, focal_length, dist, u0, v0 - 400, skew);
    cv::imshow("birdview",tmp);
}

void testBirdViewParameters(string inputUri, float scale) {
    cv::Mat img, rsImg;
    int rtn = 0;

    /******************** PROCESS ********************/
    /*** read image ***/
    rtn = vAPImread(img, inputUri, cv::IMREAD_GRAYSCALE);
    if (rtn == -1) {
        return;
    }
    cout << "INFO:\tImage shape(" << img.rows << "," << img.cols << ") " << endl;

    // std::chrono::steady_clock::time_point timer1 = std::chrono::steady_clock::now();
    /*** resize image ***/
    cv::resize(img, img, cv::Size(), scale, scale);
    cout << "INFO:\tImage shape(" << img.rows << "," << img.cols << ") " << endl;
    // cv::Rect region_of_interest= cv::Rect(0,int(img.rows*0.6),img.cols,int(img.rows*0.4));
    // cout<<"ROI "<<region_of_interest<<endl;
    // img=img(region_of_interest);

    cv::imshow("test params", img);

    cv::waitKey(0);

    alpha_degree = alpha_degree + 90;
    // gamma_degree=gamma_degree+90;

    // u0=u0+4000;
    // v0=v0+4000;
    cv::namedWindow("birdview", 1);
    cv::createTrackbar("alpha", "birdview", &alpha_degree, 360, on_trackbar, (void*)&img);
    // cv::createTrackbar("beta","birdview",&beta_degree,360,on_trackbar,(void*)&img);
    // cv::createTrackbar("gamma","birdview",&gamma_degree,360,on_trackbar,(void*)&img);
    cv::createTrackbar("focal_length", "birdview", &focal_length, 8000, on_trackbar, (void*)&img);
    cv::createTrackbar("height/sin(alpha)", "birdview", &dist, 8000, on_trackbar, (void*)&img);
    cv::createTrackbar("u0", "birdview", &u0, 8000, on_trackbar, (void*)&img);
    cv::createTrackbar("v0", "birdview", &v0, 800, on_trackbar, (void*)&img);
    cv::createTrackbar("skew", "birdview", &skew, 256000, on_trackbar, (void*)&img);
    on_trackbar(0, (void*)&img);
    cv::waitKey(0);
}

int main(int argc, char* argv[]) {
    /******************** GET PARAMETER ********************/
    if (argc < 3) {
        cout << "Missing argument!!!" << endl;
        cout << "\t[imgInputUri] [scale]" << endl;
        return -1;
    }
    inputUri = argv[1];
    float scale = stof(argv[2]);

    testBirdViewParameters(inputUri, scale);

    return 0;
}