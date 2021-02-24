#include "config.hpp"

#include <fstream>
#include <iostream>
#include <string>

#include "camera.hpp"
#include "smooth.hpp"
#include "view.hpp"

extern rld::Camera camera;
extern rld::View view;
extern rld::Gaussian gaussian;
rld::Config config;

inline static int getProperties(std::string &line, std::string &name, std::string &value) {
    line.erase(std::remove_if(line.begin(), line.end(), isspace), line.end());
    if (line[0] == '#' || line.empty()) {
        return 0;
    }
    if (line[0] == '[') {
        return -1;
    }
    std::size_t delimiterPos = line.find("=");
    if (delimiterPos == std::string::npos) {
        std::cout << "ERROR:\tCannot find '=' in line (" << line << ")" << std::endl;
        return 0;
    }

    name = line.substr(0, delimiterPos);
    value = line.substr(delimiterPos + 1);

    return 1;
}

inline static void getOption(std::ifstream &configFile, const std::string paramOpt) {
    std::string line, value;
    while (getline(configFile, line)) {
        line.erase(std::remove_if(line.begin(), line.end(), isspace), line.end());
        if (line[0] == '[') {
            value = line.substr(1, line.size() - 2);
            if (value == paramOpt) {
                return;
            }
        }
        continue;
    }
}

inline static int getCameraConfigMask(void) {
    return RLD_CAMERA_CFG_PITCH_ANGLE_MASK |
           RLD_CAMERA_CFG_YAW_ANGLE_MASK |
           RLD_CAMERA_CFG_ROLL_ANGLE_MASK |
           RLD_CAMERA_CFG_HEIGHT_MASK |
           RLD_CAMERA_CFG_FOCAL_LENGTH_MASK;
}

bool rld::getCameraConfig(std::string pathConfUrl) {
    std::string cameraConfUri = pathConfUrl + "/camera_props.conf";
    std::string line, paramOpt;
    int mask = 0;

    /********** config camera **********/
    std::cout << "INFO:\tGet camera config (" << cameraConfUri << ")" << std::endl;

    std::ifstream configFile(cameraConfUri);
    if (configFile.is_open()) {
        while (getline(configFile, line)) {
            std::string name, value;

            int rtn = getProperties(line, name, value);
            if (rtn == 1) {
                if (name == "PARAM_OPT") {
                    paramOpt = value;
                    getOption(configFile, value);
                }
                if (name == "PITCH_ANGLE") {
                    camera.pitchAngle = stof(value);
                    mask |= RLD_CAMERA_CFG_PITCH_ANGLE_MASK;

                } else if (name == "YAW_ANGLE") {
                    camera.yawAngle = stof(value);
                    mask |= RLD_CAMERA_CFG_YAW_ANGLE_MASK;

                } else if (name == "ROLL_ANGLE") {
                    camera.rollAngle = stof(value);
                    mask |= RLD_CAMERA_CFG_ROLL_ANGLE_MASK;

                } else if (name == "HEIGHT") {
                    camera.height = stof(value);
                    mask |= RLD_CAMERA_CFG_HEIGHT_MASK;

                } else if (name == "FOCAL_LENGTH") {
                    camera.focalLength = stoi(value);
                    mask |= RLD_CAMERA_CFG_FOCAL_LENGTH_MASK;
                }
            } else if (rtn == 0) {
                continue;
            } else {
                break;
            }
        }
    } else {
        std::cout << "ERROR:\tCannot open camera config file " << std::endl;
        return false;
    }
    if (mask == getCameraConfigMask()) {
        std::cout << "INFO:\t\t+ PARAM_OPT: " << paramOpt << std::endl;
        std::cout << "INFO:\t\t+ PITCH_ANGLE: " << camera.pitchAngle << std::endl;
        std::cout << "INFO:\t\t+ YAW_ANGLE: " << camera.yawAngle << std::endl;
        std::cout << "INFO:\t\t+ ROLL_ANGLE: " << camera.rollAngle << std::endl;
        std::cout << "INFO:\t\t+ HEIGHT: " << camera.height << std::endl;
        std::cout << "INFO:\t\t+ FOCAL_LENGTH: " << camera.focalLength << std::endl;
    } else {
        std::cout << "ERROR:\tMissing parameter camera config" << std::endl;
        return false;
    }

    return true;
}

inline static int getBirdViewConfigMask(void) {
    return RLD_BIRD_VIEW_CFG_U0_MASK |
           RLD_BIRD_VIEW_CFG_V0_MASK |
           RLD_BIRD_VIEW_CFG_SKEW_MASK;
}

bool rld::getCommonConfig(std::string pathConfUrl) {
    std::string commonConfUri = pathConfUrl + "/common_props.conf";
    std::string line, paramOpt;
    int birdViewCfgMask = 0;

    /********** config camera **********/
    std::cout << "INFO:\tGet common config (" << commonConfUri << ")" << std::endl;

    std::ifstream configFile(commonConfUri);
    if (configFile.is_open()) {
        config = rld::Config();
        while (getline(configFile, line)) {
            std::string name, value;
            int rtn = getProperties(line, name, value);
            if (rtn == 1) {
                /***** common config *****/
                if (name == "SCALE_RESIZE") {
                    config.common.scaleResize = stof(value);
                } else if (name == "REGION_OF_INTEREST_DX_BEGIN") {
                    config.common.rOIDxBegin = stof(value);
                } else if (name == "REGION_OF_INTEREST_DX_END") {
                    config.common.rOIDxEnd = stof(value);
                } else if (name == "REGION_OF_INTEREST_DY_BEGIN") {
                    config.common.rOIDyBegin = stof(value);
                } else if (name == "REGION_OF_INTEREST_DY_END") {
                    config.common.rOIDyEnd = stof(value);
                } else if (name == "IS_VISUALIZE_STEP_IMG") {
                    config.common.isVisualizeStepImg = stoi(value);
                } else if (name == "IS_VISUALIZE_FILTER_IMG") {
                    config.common.isVisualizeFilterImg = stoi(value);
                } else if (name == "PARAM_OPT") {
                    paramOpt = value;
                    getOption(configFile, value);
                }

                /***** bird view *****/
                else if (name == "U0") {
                    view.bvCfg.u0 = stoi(value);
                    birdViewCfgMask |= RLD_BIRD_VIEW_CFG_U0_MASK;
                } else if (name == "V0") {
                    view.bvCfg.v0 = stoi(value);
                    birdViewCfgMask |= RLD_BIRD_VIEW_CFG_V0_MASK;
                } else if (name == "SKEW") {
                    view.bvCfg.skew = stoi(value);
                    birdViewCfgMask |= RLD_BIRD_VIEW_CFG_SKEW_MASK;
                } else if (name == "MIN_ROW") {
                    view.bvCfg.minRow = stof(value);
                } else if (name == "MAX_ROW") {
                    view.bvCfg.maxRow = stof(value);
                } else if (name == "MIN_COL") {
                    view.bvCfg.minCol = stof(value);
                } else if (name == "MAX_COL") {
                    view.bvCfg.maxCol = stof(value);
                }

                /***** gaussian config *****/
                else if (name == "KERNEL_SIZE") {
                    gaussian.gausKCommon.kSize = stoi(value);
                } else if (name == "SIGMA_X") {
                    gaussian.gausKCommon.sigmaX = stof(value);
                } else if (name == "SIGMA_Y") {
                    gaussian.gausKCommon.sigmaY = stof(value);
                } else if (name == "KERNEL_SIZE_TENSOR") {
                    gaussian.gausKTensor.kSize = stoi(value);
                } else if (name == "SIGMA_X_TENSOR") {
                    gaussian.gausKTensor.sigmaX = stof(value);
                } else if (name == "SIGMA_Y_TENSOR") {
                    gaussian.gausKTensor.sigmaY = stof(value);
                }

                /***** confidence config *****/
                else if (name == "POW") {
                    config.confidenceFt.pow = stoi(value);
                } else if (name == "C") {
                    config.confidenceFt.c = stof(value);
                }
            } else if (rtn == 0) {  // error config
                continue;
            } else {  // end config
                break;
            }
        }
    } else {
        std::cout << "ERROR:\tCannot open common config file " << std::endl;
        return false;
    }
    std::cout << "INFO:\t\t+ common: " << std::endl;
    std::cout << "INFO:\t\t\t+ SCALE_RESIZE: " << config.common.scaleResize << std::endl;
    std::cout << "INFO:\t\t\t+ REGION_OF_INTEREST_DX_BEGIN: " << config.common.rOIDxBegin << std::endl;
    std::cout << "INFO:\t\t\t+ REGION_OF_INTEREST_DX_END: " << config.common.rOIDxEnd << std::endl;
    std::cout << "INFO:\t\t\t+ REGION_OF_INTEREST_DY_BEGIN: " << config.common.rOIDyBegin << std::endl;
    std::cout << "INFO:\t\t\t+ REGION_OF_INTEREST_DY_END: " << config.common.rOIDyEnd << std::endl;
    std::cout << "INFO:\t\t\t+ IS_VISUALIZE_STEP_IMG: " << config.common.isVisualizeStepImg << std::endl;
    std::cout << "INFO:\t\t\t+ IS_VISUALIZE_FILTER_IMG: " << config.common.isVisualizeFilterImg << std::endl;
    std::cout << "INFO:\t\t\t+ PARAM_OPT: " << paramOpt << std::endl;

    /* init view */
    if (birdViewCfgMask == getBirdViewConfigMask()) {
        std::cout << "INFO:\t\t+ bird view: " << std::endl;
        std::cout << "INFO:\t\t\t+ U0: " << view.bvCfg.u0 << std::endl;
        std::cout << "INFO:\t\t\t+ V0: " << view.bvCfg.v0 << std::endl;
        std::cout << "INFO:\t\t\t+ SKEW: " << view.bvCfg.skew << std::endl;
        std::cout << "INFO:\t\t\t+ MIN_ROW: " << view.bvCfg.minRow << std::endl;
        std::cout << "INFO:\t\t\t+ MAX_ROW: " << view.bvCfg.maxRow << std::endl;
        std::cout << "INFO:\t\t\t+ MIN_COL: " << view.bvCfg.minCol << std::endl;
        std::cout << "INFO:\t\t\t+ MAX_COL: " << view.bvCfg.maxCol << std::endl;

        /* init bird view mask */
        view.calBirdViewMask();
    } else {
        std::cout << "ERROR:\tMissing parameter bird view config" << std::endl;
        return false;
    }

    std::cout << "INFO:\t\t+ gaussian: " << std::endl;
    std::cout << "INFO:\t\t\t+ KERNEL_SIZE: " << gaussian.gausKCommon.kSize << std::endl;
    std::cout << "INFO:\t\t\t+ SIGMA_X: " << gaussian.gausKCommon.sigmaX << std::endl;
    std::cout << "INFO:\t\t\t+ SIGMA_Y: " << gaussian.gausKCommon.sigmaY << std::endl;
    std::cout << "INFO:\t\t\t+ KERNEL_SIZE_TENSOR: " << gaussian.gausKTensor.kSize << std::endl;
    std::cout << "INFO:\t\t\t+ SIGMA_X_TENSOR: " << gaussian.gausKTensor.sigmaX << std::endl;
    std::cout << "INFO:\t\t\t+ SIGMA_Y_TENSOR: " << gaussian.gausKTensor.sigmaY << std::endl;
    std::cout << "INFO:\t\t+ confidence filter: " << std::endl;
    std::cout << "INFO:\t\t\t+ POW: " << config.confidenceFt.pow << std::endl;
    std::cout << "INFO:\t\t\t+ C: " << config.confidenceFt.c << std::endl;

    return true;
}

bool rld::getConfig(std::string pathConfUrl) {
    return getCameraConfig(pathConfUrl) && getCommonConfig(pathConfUrl);
}

rld::Config::Config() {
    /***** common *****/
    this->common.scaleResize = 1;
    this->common.rOIDxBegin = 1;
    this->common.rOIDxEnd = 1;
    this->common.rOIDyBegin = 1;
    this->common.rOIDyEnd = 1;
    this->common.isVisualizeStepImg = 0;
    this->common.isVisualizeFilterImg = 0;

    /***** confidence *****/
    this->confidenceFt.pow = 2;
    this->confidenceFt.c = 0.001;
}

rld::Config::~Config() {}