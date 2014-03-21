Recognizing Panoramas in Video
Stanford CS 231A Final Project
Built on top of OpenCV Stitching Module
_____________________________________________________________________

Program file: video_stitching_detailed.cpp

Automatically recognizes panoramic scenes in a video and tries to 
produce a panorama from each detected scene.

Runs in ~20 mins for a 15-second video with 2 panoramas...not exactly
fast at this point.

Built on top of OpenCV's Stitching module example code 
stitching_detailed.cpp

Paper describing program technique included: ProjectPaper.pdf
_____________________________________________________________________

Example video input and output panoramas included:
garden.avi -> garden1.jpg
quad.avi   -> quad1.jpg , quad2.jpg
_____________________________________________________________________

Default parameters usually work well, but here are some tips:
If the panorama is large, use "--warp cylindrical"
If the panoramic scenes/segments are too short, try:
"--match_conf 0.8 --conf_thresh 0.6" or lower values if needed
_____________________________________________________________________

DISCLAIMER:
This work is very un-polished, but it should run reasonably well on 
any standard personal computer.  The goal is to simply show how 
this functionality of creating panoramas from video can be done by 
adding some steps to what OpenCV's Stitching module already does.
_____________________________________________________________________

Uses C++11 standard and OpenCV version 2.4.8

Build using "cmake ." to create Makefile, then "make" to build
exectuable.
