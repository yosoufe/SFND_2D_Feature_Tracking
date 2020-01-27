# Report

- [Introduction](#Introduction)
- [MP1 - Data Buffer Optimization](#MP1)
- [MP2 - Keypoint Detection](#MP2)
- [MP3 - Keypoint Removal](#MP3)

<a name="Introduction" />

## Introduction
This is a report to cover the PROJECT SPECIFICATION for 2nd project of Sensor Fusion Nanodegree, **2D Feature Tracking**.


<a name="MP1" />

## MP1 - Data Buffer Optimization
This is required for long data stream of images. The following code section would implement a std::vector that 
its size would never get larger than the specified values `dataBufferSize`, here 2. When the size is smaller than `dataBufferSize` frame is being added normally. But when the buffer full, first the frames in the buffer are copied 
to the location with one index less and then the new frame is being copied two the end of the buffer.
This is implemented in `src/MidTermProject_Camera_Student.cpp` file.
```c++
DataFrame frame;
frame.cameraImg = imgGray;
if (dataBuffer.size() < dataBufferSize)
{
    dataBuffer.push_back(frame);
} else
{
    for ( auto it = dataBuffer.begin() ; it != dataBuffer.end()-1 ; it++ )
    {
        *it = *(it+1);
    }
    dataBuffer[dataBufferSize - 1] = frame;
}
``` 
The benefit is that it would follow the same api of normal `std::vector` it would not be
required to change the rest of the code to access the data. While the disadvantage is that
if the `dataBufferSize` would be large, then a lot of data copy would be required which would 
not be ideal. In this case, 
[Circular Buffer from Boost library](https://www.boost.org/doc/libs/1_61_0/doc/html/circular_buffer.html) 
is highly recommended.

<a name="MP2" />

## MP2 - Keypoint Detection
Multiple detectors have been integrated from OpenCV including 
SHITOMASI, HARRIS, FAST, BRISK, ORB, AKAZE, SIFT, ORB_CUDA and FAST_CUDA. 
Of course the CUDA ones would only work of the project is compiled with CUDA option
enabled. Please checkout the readme file on how to enable it. The following function 
is creating multiple detectors based on the given string and also extracts the keypoints.
This is implemented in `src/matching2D_Student.cpp` file.
```c++
void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints,
                        cv::Mat &img,
                        std::string detectorType,
                        bool bVis)
{
    cv::Ptr<cv::FeatureDetector> detector = nullptr;
    if (detectorType.compare("FAST") == 0)
    {
        int threshold = 30;                                                              // difference between intensity of the central pixel and pixels of a circle around this pixel
        bool bNMS = true;                                                                // perform non-maxima suppression on keypoints
        cv::FastFeatureDetector::DetectorType type = cv::FastFeatureDetector::TYPE_9_16; // TYPE_9_16, TYPE_7_12, TYPE_5_8
        detector = cv::FastFeatureDetector::create(threshold, bNMS, type);
    // https://docs.opencv.org/4.1.0/d5/d51/group__features2d__main.html
    // BRISK, ORB, AKAZE, and SIFT
    } else if (detectorType.compare("BRISK") == 0){
        detector = cv::BRISK::create();
    } else if (detectorType.compare("ORB") == 0){
        detector = cv::ORB::create();
    } else if (detectorType.compare("AKAZE") == 0){
        detector = cv::AKAZE::create();
    } else if (detectorType.compare("SIFT") == 0){
        detector = cv::xfeatures2d::SIFT::create();
    } 
    if (detector) 
    {
        detector->detect(img, keypoints);
        return;
    }

#if WITH_CUDA
    if(detectorType.compare("ORB_CUDA") == 0)
    {
        detector = cv::cuda::ORB::create();
    } 
    else if(detectorType.compare("FAST_CUDA") == 0)
    {
        int threshold = 30;                                                              // difference between intensity of the central pixel and pixels of a circle around this pixel
        bool bNMS = true;                                                                // perform non-maxima suppression on keypoints
        cv::FastFeatureDetector::DetectorType type = cv::FastFeatureDetector::TYPE_9_16; // TYPE_9_16, TYPE_7_12, TYPE_5_8
        detector = cv::cuda::FastFeatureDetector::create(threshold, bNMS, type);
    }
    if (detector) 
    {
        cv::cuda::GpuMat imageGpu;
        imageGpu.upload(img);
        detector->detect(imageGpu, keypoints);
    }
#endif // WITH_CUDA
}
```
HARRIS and SHITOMASI are defined in other functions called `detKeypointsHarris` and `detKeypointsShiTomasi` in the same file.

<a name="MP3" />

## MP3 - Keypoint Removal

For debug purpose and also as specified in this project specification it is nice to just focus on the keypoints that 
are on the preceding vehicle. This is done by using `cv::Rect` type and its `cv::Rect::contains` member function 
and removing the points that are not in that list.

This feature is dependent on the boolean flag `bFocusOnVehicle` which can be assigned with `-f` argument when executing the 
project executable.

This is implemented in `src/MidTermProject_Camera_Student.cpp` file.
```c++
// only keep keypoints on the preceding vehicle
if (bFocusOnVehicle)
{
    cv::Rect vehicleRect(535, 180, 180, 150);
    std::vector<cv::KeyPoint> kp_on_preceding_car;
    for (auto & kpt : keypoints)
    {
        if (vehicleRect.contains(kpt.pt))
            kp_on_preceding_car.push_back(kpt);
    }
    keypoints = kp_on_preceding_car;
    cout << "Number of Keypoints on Preceding Vehicle: " << keypoints.size() << endl;
}
```