#ifndef RLD_CONFIG_HPP
#define RLD_CONFIG_HPP

#include <string>

namespace rld {

/***** camera config mask *****/
#define RLD_CAMERA_CFG_PITCH_ANGLE_MASK 0x1
#define RLD_CAMERA_CFG_YAW_ANGLE_MASK 0x2
#define RLD_CAMERA_CFG_ROLL_ANGLE_MASK 0x4
#define RLD_CAMERA_CFG_HEIGHT_MASK 0x8
#define RLD_CAMERA_CFG_FOCAL_LENGTH_MASK 0x10

/***** bird view config mask *****/
#define RLD_BIRD_VIEW_CFG_U0_MASK 0x1
#define RLD_BIRD_VIEW_CFG_V0_MASK 0x2
#define RLD_BIRD_VIEW_CFG_SKEW_MASK 0x4

struct rldCommonCfg {
    float scaleResize;
    float rOIDxBegin;
    float rOIDxEnd;
    float rOIDyBegin;
    float rOIDyEnd;
    bool isVisualizeStepImg;
    bool isVisualizeFilterImg;
};

struct rldConfidenceFilterCfg {
    int pow;
    float c;
};

bool getCameraConfig(std::string pathConfUrl);
bool getCommonConfig(std::string pathConfUrl);
bool getConfig(std::string pathConfUrl);

class Config {
   public:
    rldCommonCfg common;
    rldConfidenceFilterCfg confidenceFt;

    Config();
    ~Config();
};

}  // namespace rld

#endif