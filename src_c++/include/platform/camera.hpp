#ifndef RLD_CAMERA_HPP
#define RLD_CAMERA_HPP

#include <opencv2/core/mat.hpp>

namespace rld {

class Camera {
   public:
    float pitchAngle;
    float yawAngle;
    float rollAngle;
    float height;
    int focalLength;

    Camera();
    ~Camera();
};

}  // namespace rld

#endif