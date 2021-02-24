cmake_minimum_required(VERSION 3.10.2)

project(lane_detection LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wpedantic -Wcast-align -Wunused -Wpointer-arith -Wno-missing-braces -g")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(Threads REQUIRED)

include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER AND (USE_CUDA STREQUAL "ON"))
	enable_language(CUDA)
	find_package(CUDA REQUIRED)
	include_directories( ${CUDA_INCLUDE_DIRS})
	add_definitions(-DUSE_CUDA)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_CUDA")
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DUSE_CUDA")
endif ()

find_package( OpenCV REQUIRED PATHS /usr/local/ NO_DEFAULT_PATH)
include_directories( ${OpenCV_INCLUDE_DIRS} )

find_package (Eigen3 3.3 REQUIRED )

include_directories( include/common include/platform include/util include/kernel include )

file(GLOB SrcMain		src/*.cpp )
file(GLOB SrcUtil		src/util/*.cpp )
file(GLOB SrcCommon		src/common/*.cpp )
file(GLOB SrcPlatform	src/platform/*.cpp )

if (CMAKE_CUDA_COMPILER AND (USE_CUDA STREQUAL "ON"))
	file(GLOB SrcKernel	src/kernel/*.cu )
else()
	list(REMOVE_ITEM SrcPlatform "${PROJECT_SOURCE_DIR}/src/platform/gpu.cpp")
	list(REMOVE_ITEM SrcMain "${PROJECT_SOURCE_DIR}/src/cuda_lane_detection.cpp")
endif ()

add_executable( lane_detection ${SrcUtil} ${SrcCommon} ${SrcPlatform} ${SrcMain} ${SrcKernel} )

if(CMAKE_CUDA_COMPILER AND (USE_CUDA STREQUAL "ON"))
	target_link_libraries( lane_detection ${OpenCV_LIBS} ${CUDA_LIBS} Eigen3::Eigen Threads::Threads )
else ()
	target_link_libraries( lane_detection ${OpenCV_LIBS} Eigen3::Eigen Threads::Threads )
endif ()




