cmake_minimum_required(VERSION 3.10.2)
project(get_camera_params LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wpedantic -Wcast-align -Wunused -Wpointer-arith -Wno-missing-braces -g")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

file(GLOB SrcMain		*.cpp )

add_executable( get_camera_params ${SrcMain})
target_link_libraries( get_camera_params ${OpenCV_LIBS})





