# SFND 2D Feature Tracking

<img src="images/keypoints.png" width="820" height="248" />

The idea of the camera course is to build a collision detection system - that's the overall goal for the Final Project. As a preparation for this, we build the feature tracking part and test various detector / descriptor combinations to see which ones perform best. This project consists of four parts:

* First, loading images, setting up data structures and putting everything into a ring buffer to optimize memory load. 
* Then, integrating several keypoint detectors such as **HARRIS**, **FAST**, **BRISK** and **SIFT** and comparing them with regard to number of keypoints and speed. 
* The next part, descriptor extraction and matching using **brute force** and also the **FLANN** approach. 
* The last part, once the code framework is complete, various algorithms in different combinations are being tested and they are compared with regard to some performance measures.  

## Dependencies for Running Locally
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)
* for argument parsing I am using [argparse in this github repo](https://github.com/cofyc/argparse)
   which is already included as a submodule here.

## Basic Build Instructions

1. Clone this repo and submodule [argparse](https://github.com/cofyc/argparse).

   ```
   git clone https://github.com/yosoufe/SFND_2D_Feature_Tracking.git
   cd SFND_2D_Feature_Tracking
   git submodule update --init --recursive
   ```

2. Make a build directory in the top level directory: `mkdir build && cd build`
3. Compile without GPU Support: `cmake .. && make`

      * With GPU support: `cmake -DWITH_CUDA=ON .. && make`:
         
         In case of GPU supports the extra following items would be available: 
            
          * detectors: **ORB_CUDA** and **FAST_CUDA**
          * matcher: **MAT_BF_CUDA**
          * descriptor: **ORB_CUDA**


4. Run it:

   * The executable accepts multiple optional arguments to define multiple variables.
   * use `./2D_feature_tracking -h` for help. It would create the following output:

      ```
      $ ./2D_feature_tracking -h
      Usage: ./2D_feature_tracking [args]
      For example: ./2D_feature_tracking --detector_type=BRISK --matcher_type=MAT_FLANN --descriptor_type=DES_BINARY --selector_type=SEL_KNN -f -q

      Explores different 2d keypoint detector, descriptor and matching

          -h, --help                show this help message and exit

      Optional Arguments: 
          --detector_type=<str>     detector type, options: SHITOMASI, HARRIS, FAST, BRISK, ORB, AKAZE, and SIFT
              if compiled (WITH_CUDA on): ORB_CUDA, FAST_CUDA
              default: SHITOMASI
          --matcher_type=<str>      matcher type, options: MAT_BF, MAT_FLANN,
              if compiled (WITH_CUDA on): MAT_BF_CUDA
              default: MAT_BF
          --descriptor_type=<str>   descriptor type, options: BRISK BRIEF, ORB, FREAK, AKAZE, SIFT
              if compiled (WITH_CUDA on): ORB_CUDA
              default: BRISK
          --selector_type=<str>     selector type, options: SEL_NN, SEL_KNN
              default: SEL_NN
          -f, --focus_on_vehicle    To focus on only keypoints that are on the preceding vehicle.
          -l, --limit_keypoints     To limit the number of keypoints to maximum 50 keypoints.
          -q, --quiet               If this flag is chosen no image would be shown. Good for performance measurement
      ```
