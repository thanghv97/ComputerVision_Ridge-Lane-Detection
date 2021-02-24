#include "camera.hpp"

#include "include.hpp"

rld::Camera camera = rld::Camera();

rld::Camera::Camera() {
    this->pitchAngle = 0;
    this->yawAngle = 0;
    this->rollAngle = 0;
    this->height = 0;
    this->focalLength = 0;
}

rld::Camera::~Camera() {}