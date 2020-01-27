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
is creating multiple detectors based on the given string
```c++
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor = nullptr;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    // BRIEF, ORB, FREAK, AKAZE, SIFT
    else if (descriptorType.compare("BRIEF") == 0)
    {
        int bytes=32;               // length of the descriptor in bytes, valid values are: 16, 32 (default) or 64 .
        bool use_orientation=false; // sample patterns using keypoints orientation, disabled by default.
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create(bytes, use_orientation);
        
    }
    else if (descriptorType.compare("ORB") == 0)
    {
        extractor = cv::ORB::create();
    } 
    else if (descriptorType.compare("FREAK") == 0)
    {
        extractor = cv::xfeatures2d::FREAK::create();
    }
    else if (descriptorType.compare("AKAZE") == 0)
    {
        extractor = cv::AKAZE::create();
    }
    else if (descriptorType.compare("SIFT") == 0)
    {
        extractor = cv::xfeatures2d::SIFT::create();
    }

    if (extractor) 
    {
        // perform feature description
        extractor->compute(img, keypoints, descriptors);
        return;
    }
#if WITH_CUDA
    if(descriptorType.compare("ORB_CUDA") == 0)
    {
        extractor = cv::cuda::ORB::create();
    }
    if (extractor) 
    {
        cv::cuda::GpuMat imageGpu;
        cv::cuda::GpuMat d_descriptors;
        imageGpu.upload(img);
        extractor->compute(imageGpu, keypoints, d_descriptors);
        d_descriptors.download(descriptors);
        
    }
#endif // 
```


<a name="MP3" />

## MP3 - Keypoint Removal