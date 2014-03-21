cmake_minimum_required(VERSION 2.8)
project( video_stitching_detailed )
find_package( OpenCV REQUIRED )
add_executable( video_stitching_detailed video_stitching_detailed.cpp )

set(CMAKE_CXX_FLAGS "-g -std=c++0x")

target_link_libraries( video_stitching_detailed ${OpenCV_LIBS} )
