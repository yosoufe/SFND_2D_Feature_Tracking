# Report

## Content
- [Introduction](#Introduction)
- [Data Buffer](#DataBuffer)
    - [MP1 - Data Buffer Optimization](#MP1)
- [Keypoints](#Keypoints)
    - [MP2 - Keypoint Detection](#MP2)
    - [MP3 - Keypoint Removal](#MP3)
- [Descriptors](#Descriptors)
    - [MP4 - Keypoint Descriptors](#MP4)
    - [MP5 - Descriptor Matching](#MP5)
    - [MP6 - Descriptor Distance Ratio](#MP6)
- [Performance](#Performance)
    - [MP7 - Performance Evaluation 1](#MP7)
    - [MP8 - Performance Evaluation 2](#MP8)
    - [MP9 - Performance Evaluation 3](#MP9)

<a name="Introduction" />

## Introduction
This is a report to cover the PROJECT SPECIFICATION for 2nd project of Sensor Fusion Nanodegree, **2D Feature Tracking**.


<a name="DataBuffer" />

## Data Buffer

<a name="MP1" />

### MP1 - Data Buffer Optimization
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

<a name="Keypoints" />

## Keypoints

<a name="MP2" />

### MP2 - Keypoint Detection
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

### MP3 - Keypoint Removal

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

<a name="Descriptors" />

## Descriptors

<a name="MP4" />

### MP4 - Keypoint Descriptors

Different keypoint descriptors such as BRISK BRIEF, ORB, FREAK, AKAZE, SIFT and ORB_CUDA are integrated in the following function in `src/matching2D_Student.cpp`. Of course the CUDA ones would only work of the project is compiled with CUDA option
enabled.

```c++
void descKeypoints(vector<cv::KeyPoint> &keypoints, 
                   cv::Mat &img, 
                   cv::Mat &descriptors, 
                   string descriptorType)
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
#endif // WITH_CUDA
}
```

<a name="MP5" />

### MP5 - Descriptor Matching

This is integrated in `matchDescriptors` function in `src/matching2D_Student.cpp`. 
FLANN matching, K-nearest neighbor and Brute force matcher are integrated and also 
with CUDA support if it is available.


<a name="MP6" />

### MP6 - Descriptor Distance Ratio

This is integrated in as a part of `matchDescriptors` function in `src/matching2D_Student.cpp`
as the follow.

```c++
std::vector< std::vector<cv::DMatch> > knn_matches;
matcher->knnMatch( descSource, descRef, knn_matches, 2 );
// filter matches using descriptor distance ratio test
double minDescDistRatio = 0.8;
for (auto it = knn_matches.begin(); it != knn_matches.end(); ++it)
{
    if ((*it)[0].distance < minDescDistRatio * (*it)[1].distance)
    {
        matches.push_back((*it)[0]);
    }
}
```

<a name="Performance" />

## Performance

I am using the python script in `src/runner.py` to run the executable with different 
arguments and parse the output to provide the following tables. It outputs commands that 
can be copied and pasted in Google Sheets to create proper tables then 
[this Markdon Table Maker](https://gsuite.google.com/marketplace/app/markdowntablemaker/46507245362?pann=cwsdp&hl=en) 
is used to output markdown. 

All the following numbers are being calculated on my local machine rather than Udacity's workspace.

<a name="MP7" />

### MP7 - Performance Evaluation 1

The following table shows the **number of keypoints on the preceding vehicles** 
for different detector for all 10 images:

|  **Detector** | **Image 0** | **Image 1** | **Image 2** | **Image 3** | **Image 4** | **Image 5** | **Image 6** | **Image 7** | **Image 8** | **Image 9** |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  **SHITOMASI** | 125 | 118 | 123 | 120 | 120 | 113 | 114 | 123 | 111 | 112 |
|  **HARRIS** | 17 | 14 | 18 | 21 | 26 | 43 | 18 | 31 | 26 | 34 |
|  **FAST** | 149 | 152 | 150 | 155 | 149 | 149 | 156 | 150 | 138 | 143 |
|  **FAST_CUDA** | 104 | 107 | 110 | 113 | 109 | 115 | 113 | 121 | 95 | 116 |
|  **BRISK** | 264 | 282 | 282 | 277 | 297 | 279 | 289 | 272 | 267 | 254 |
|  **ORB** | 92 | 102 | 106 | 113 | 109 | 125 | 130 | 129 | 127 | 128 |
|  **ORB_CUDA** | 95 | 107 | 109 | 121 | 103 | 126 | 126 | 128 | 133 | 130 |
|  **AKAZE** | 166 | 157 | 161 | 155 | 163 | 164 | 173 | 175 | 177 | 179 |
|  **SIFT** | 138 | 132 | 124 | 137 | 134 | 140 | 137 | 148 | 159 | 137 |

<a name="MP8" />

### MP8 - Performance Evaluation 2

The following table **counts the number of matched keypoints** for all 10 images using all possible combinations of detectors and descriptors. In the matching step, the BF approach is used with the descriptor distance ratio set to 0.8.

|  Detector | Descriptor | Images 0-1 | Images 1-2 | Images 2-3 | Images 3-4 | Images 4-5 | Images 5-6 | Images 6-7 | Images 7-8 | Images 8-9 |
| --- | --- | --- | :--- | --- | :--- | --- | :--- | --- | :--- | --- |
|  SHITOMASI | BRISK | 95 | 88 | 80 | 90 | 82 | 79 | 85 | 86 | 82 |
|  SHITOMASI | BRIEF | 115 | 111 | 104 | 101 | 102 | 102 | 100 | 109 | 100 |
|  SHITOMASI | ORB | 106 | 102 | 99 | 102 | 103 | 97 | 98 | 104 | 97 |
|  SHITOMASI | ORB_CUDA | 100 | 92 | 85 | 94 | 90 | 89 | 89 | 93 | 88 |
|  SHITOMASI | FREAK | 86 | 90 | 86 | 88 | 86 | 80 | 81 | 86 | 85 |
|  SHITOMASI | AKAZE | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
|  SHITOMASI | SIFT | 112 | 109 | 104 | 103 | 99 | 101 | 96 | 106 | 97 |
|  HARRIS | BRISK | 12 | 10 | 14 | 15 | 16 | 16 | 15 | 23 | 21 |
|  HARRIS | BRIEF | 14 | 11 | 15 | 20 | 24 | 26 | 16 | 24 | 23 |
|  HARRIS | ORB | 12 | 12 | 15 | 18 | 24 | 20 | 15 | 24 | 22 |
|  HARRIS | ORB_CUDA | 12 | 13 | 16 | 18 | 23 | 18 | 13 | 24 | 21 |
|  HARRIS | FREAK | 13 | 13 | 15 | 15 | 17 | 20 | 12 | 21 | 18 |
|  HARRIS | AKAZE | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
|  HARRIS | SIFT | 14 | 11 | 16 | 19 | 22 | 22 | 13 | 24 | 22 |
|  FAST | BRISK | 97 | 104 | 101 | 98 | 85 | 107 | 107 | 100 | 100 |
|  FAST | BRIEF | 119 | 130 | 118 | 126 | 108 | 123 | 131 | 125 | 119 |
|  FAST | ORB | 118 | 123 | 112 | 126 | 106 | 122 | 122 | 123 | 119 |
|  FAST | ORB_CUDA | 112 | 114 | 106 | 118 | 97 | 117 | 117 | 118 | 111 |
|  FAST | FREAK | 98 | 99 | 91 | 98 | 85 | 99 | 102 | 101 | 105 |
|  FAST | AKAZE | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
|  FAST | SIFT | 118 | 123 | 110 | 119 | 114 | 119 | 123 | 117 | 103 |
|  FAST_CUDA | BRISK | 52 | 61 | 63 | 48 | 48 | 64 | 70 | 61 | 55 |
|  FAST_CUDA | BRIEF | 71 | 68 | 71 | 68 | 66 | 84 | 80 | 79 | 76 |
|  FAST_CUDA | ORB | 71 | 76 | 62 | 61 | 59 | 87 | 90 | 91 | 71 |
|  FAST_CUDA | ORB_CUDA | 59 | 64 | 75 | 76 | 60 | 83 | 85 | 85 | 72 |
|  FAST_CUDA | FREAK | 66 | 67 | 63 | 66 | 52 | 68 | 70 | 61 | 61 |
|  FAST_CUDA | AKAZE | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
|  FAST_CUDA | SIFT | 72 | 71 | 71 | 76 | 75 | 89 | 72 | 77 | 70 |
|  BRISK | BRISK | 171 | 176 | 157 | 176 | 174 | 187 | 173 | 172 | 184 |
|  BRISK | BRIEF | 178 | 205 | 185 | 180 | 182 | 196 | 206 | 189 | 183 |
|  BRISK | ORB | 162 | 175 | 158 | 167 | 160 | 183 | 168 | 170 | 173 |
|  BRISK | ORB_CUDA | 138 | 152 | 129 | 138 | 118 | 155 | 140 | 148 | 149 |
|  BRISK | FREAK | 160 | 177 | 155 | 173 | 163 | 182 | 169 | 178 | 167 |
|  BRISK | AKAZE | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
|  BRISK | SIFT | 182 | 193 | 170 | 183 | 172 | 195 | 194 | 176 | 183 |
|  ORB | BRISK | 73 | 74 | 79 | 85 | 79 | 92 | 90 | 88 | 91 |
|  ORB | BRIEF | 49 | 43 | 45 | 59 | 53 | 78 | 68 | 84 | 66 |
|  ORB | ORB | 67 | 70 | 72 | 84 | 91 | 101 | 92 | 93 | 93 |
|  ORB | ORB_CUDA | 66 | 71 | 71 | 73 | 78 | 86 | 85 | 80 | 91 |
|  ORB | FREAK | 42 | 36 | 44 | 47 | 44 | 51 | 52 | 48 | 56 |
|  ORB | AKAZE | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
|  ORB | SIFT | 67 | 79 | 78 | 79 | 82 | 95 | 95 | 94 | 94 |
|  ORB_CUDA | BRISK | 70 | 79 | 81 | 87 | 80 | 97 | 88 | 87 | 94 |
|  ORB_CUDA | BRIEF | 49 | 51 | 48 | 61 | 54 | 82 | 73 | 68 | 65 |
|  ORB_CUDA | ORB | 72 | 83 | 73 | 86 | 77 | 98 | 79 | 91 | 102 |
|  ORB_CUDA | ORB_CUDA | 67 | 81 | 67 | 77 | 75 | 89 | 76 | 86 | 96 |
|  ORB_CUDA | FREAK | 44 | 39 | 50 | 50 | 42 | 49 | 47 | 50 | 59 |
|  ORB_CUDA | AKAZE | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
|  ORB_CUDA | SIFT | 69 | 77 | 73 | 85 | 78 | 100 | 95 | 97 | 102 |
|  AKAZE | BRISK | 137 | 125 | 129 | 129 | 131 | 132 | 142 | 146 | 144 |
|  AKAZE | BRIEF | 141 | 134 | 131 | 130 | 134 | 146 | 150 | 148 | 152 |
|  AKAZE | ORB | 131 | 129 | 127 | 117 | 130 | 131 | 137 | 135 | 145 |
|  AKAZE | ORB_CUDA | 122 | 118 | 110 | 108 | 111 | 120 | 126 | 126 | 129 |
|  AKAZE | FREAK | 126 | 129 | 127 | 121 | 122 | 133 | 144 | 147 | 138 |
|  AKAZE | AKAZE | 138 | 138 | 133 | 127 | 129 | 146 | 147 | 151 | 150 |
|  AKAZE | SIFT | 134 | 134 | 130 | 136 | 137 | 147 | 147 | 154 | 151 |
|  SIFT | BRISK | 64 | 66 | 62 | 66 | 59 | 64 | 64 | 67 | 80 |
|  SIFT | BRIEF | 86 | 78 | 76 | 85 | 69 | 74 | 76 | 70 | 88 |
|  SIFT | ORB | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
|  SIFT | ORB_CUDA | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
|  SIFT | FREAK | 65 | 72 | 64 | 66 | 59 | 59 | 64 | 65 | 79 |
|  SIFT | AKAZE | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
|  SIFT | SIFT | 82 | 81 | 85 | 93 | 90 | 81 | 82 | 102 | 104 |

<a name="MP9" />

### MP9 - Performance Evaluation 3

The following table describes **the time it takes for keypoint detection and descriptor extraction in seconds**.

|  Detector | Descriptor | Average | Image 0 | Image 1 | Image 2 | Image 3 | Image 4 | Image 5 | Image 6 | Image 7 | Image 8 | Image 9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  SHITOMASI | BRISK | 183.618933 | 190.1942 | 181.97511 | 184.848 | 182.79324 | 182.22148 | 183.61576 | 181.98255 | 183.62704 | 182.429 | 182.50295 |
|  SHITOMASI | BRIEF | 10.9344779 | 14.070853 | 10.736296 | 10.754014 | 10.656113 | 10.489483 | 10.236746 | 10.857348 | 10.574154 | 10.446167 | 10.523605 |
|  SHITOMASI | ORB | 10.9477414 | 12.51588 | 10.93001 | 10.71819 | 10.844198 | 10.60636 | 10.741804 | 11.005596 | 10.651947 | 10.666589 | 10.79684 |
|  SHITOMASI | ORB_CUDA | 37.123147 | 298.2148 | 10.674465 | 7.838598 | 7.694929 | 8.044426 | 7.727307 | 7.650336 | 7.786441 | 7.858221 | 7.741947 |
|  SHITOMASI | FREAK | 30.984526 | 39.5725 | 33.01185 | 30.006 | 29.90168 | 29.98566 | 29.82954 | 29.31673 | 29.6712 | 29.44553 | 29.10457 |
|  SHITOMASI | AKAZE | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
|  HARRIS | BRISK | 187.205913 | 188.3178 | 183.139 | 184.0576 | 183.6178 | 184.7369 | 191.8455 | 184.83273 | 188.4826 | 188.2674 | 194.7618 |
|  HARRIS | BRIEF | 12.7622417 | 16.087033 | 11.194509 | 11.048444 | 11.066155 | 11.310727 | 18.563216 | 10.793902 | 12.082187 | 11.218928 | 14.257316 |
|  HARRIS | ORB | 12.4761546 | 12.897387 | 11.092602 | 11.006021 | 10.877336 | 11.328504 | 18.938118 | 11.304484 | 11.907875 | 11.395467 | 14.013752 |
|  HARRIS | ORB_CUDA | 35.6140959 | 269.8378 | 10.844571 | 8.113891 | 8.089557 | 8.163111 | 15.308913 | 7.580367 | 8.814414 | 8.311227 | 11.077108 |
|  HARRIS | FREAK | 34.922916 | 70.7749 | 32.9624 | 29.72874 | 29.98808 | 29.51645 | 35.9541 | 28.63641 | 29.87715 | 29.27383 | 32.5171 |
|  HARRIS | AKAZE | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
|  FAST | BRISK | 178.5452592 | 177.713315 | 177.770426 | 177.77212 | 180.302046 | 181.874705 | 181.883659 | 180.017638 | 177.78493 | 176.216889 | 174.116864 |
|  FAST | BRIEF | 1.2896349 | 3.14594 | 1.702264 | 1.358493 | 0.970545 | 0.966703 | 0.934371 | 0.983778 | 0.965982 | 0.939949 | 0.928324 |
|  FAST | ORB | 2.1953414 | 5.01512 | 3.01779 | 3.2731 | 2.62049 | 1.62149 | 1.255513 | 1.247411 | 1.509543 | 1.186361 | 1.206596 |
|  FAST | ORB_CUDA | 25.6925056 | 246.56658 | 1.213834 | 1.185775 | 1.142272 | 1.127435 | 1.209709 | 1.112022 | 1.11938 | 1.132114 | 1.115935 |
|  FAST | FREAK | 23.7580367 | 27.174747 | 25.491206 | 23.534304 | 23.057706 | 22.843932 | 22.750109 | 23.459857 | 22.867949 | 23.443357 | 22.9572 |
|  FAST | AKAZE | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
|  FAST_CUDA | BRISK | 192.6439273 | 332.261 | 177.933694 | 176.338808 | 178.112524 | 176.400447 | 177.183491 | 177.203213 | 177.23726 | 177.475294 | 176.293542 |
|  FAST_CUDA | BRIEF | 15.058769 | 141.7679 | 1.457381 | 1.059315 | 1.073808 | 0.868882 | 0.841434 | 0.878895 | 0.875319 | 0.813041 | 0.951715 |
|  FAST_CUDA | ORB | 16.284382 | 151.847546 | 1.420607 | 1.267066 | 1.265492 | 1.143368 | 1.209445 | 1.17515 | 1.149916 | 1.221175 | 1.144055 |
|  FAST_CUDA | ORB_CUDA | 22.6256988 | 216.3753 | 1.161929 | 1.092087 | 1.06892 | 1.063202 | 1.098785 | 1.075938 | 1.069034 | 1.159353 | 1.09244 |
|  FAST_CUDA | FREAK | 37.6654171 | 170.9953 | 23.323835 | 23.252432 | 22.837184 | 22.912038 | 22.837838 | 22.534097 | 22.832508 | 22.354263 | 22.774676 |
|  FAST_CUDA | AKAZE | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
|  BRISK | BRISK | 389.5905 | 384.223 | 380.291 | 390.865 | 393.958 | 395.863 | 386.933 | 394.365 | 391.938 | 388.749 | 388.72 |
|  BRISK | BRIEF | 212.4310155 | 213.42647 | 211.086031 | 212.497113 | 211.810822 | 213.884846 | 229.24886 | 208.347704 | 208.713412 | 209.083507 | 206.21139 |
|  BRISK | ORB | 215.206298 | 219.36269 | 216.00026 | 214.54275 | 214.96506 | 216.32368 | 209.72 | 215.49985 | 214.96561 | 215.86203 | 214.82105 |
|  BRISK | ORB_CUDA | 237.973945 | 489.628 | 205.98806 | 209.19834 | 211.54251 | 210.48252 | 206.60348 | 208.99906 | 208.4887 | 209.81912 | 218.98966 |
|  BRISK | FREAK | 233.44158 | 234.0675 | 233.8747 | 232.9209 | 232.4973 | 233.684 | 235.2053 | 235.3419 | 232.9144 | 229.8691 | 234.0407 |
|  BRISK | AKAZE | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
|  ORB | BRISK | 207.311935 | 399.898 | 188.991 | 188.29004 | 185.5455 | 185.56762 | 187.18269 | 185.19937 | 186.26905 | 181.69386 | 184.48222 |
|  ORB | BRIEF | 21.9480581 | 167.582483 | 6.551724 | 5.751017 | 5.823598 | 5.534148 | 5.471764 | 5.229306 | 5.73265 | 5.819132 | 5.984759 |
|  ORB | ORB | 24.633553 | 164.84336 | 10.02997 | 8.73371 | 9.11479 | 9.09757 | 8.58317 | 9.17391 | 8.76298 | 9.07664 | 8.91943 |
|  ORB | ORB_CUDA | 36.576323 | 306.21 | 7.49793 | 6.79713 | 6.75257 | 6.27927 | 6.58482 | 6.34472 | 6.32916 | 6.56621 | 6.40142 |
|  ORB | FREAK | 42.578166 | 175.6881 | 27.91328 | 27.78881 | 28.16879 | 27.80545 | 27.52315 | 27.55005 | 27.11062 | 27.72582 | 28.50759 |
|  ORB | AKAZE | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
|  ORB_CUDA | BRISK | 210.549878 | 446.053 | 180.19293 | 182.28146 | 184.05222 | 183.32949 | 196.48195 | 182.67707 | 183.42937 | 183.91363 | 183.08766 |
|  ORB_CUDA | BRIEF | 26.9896736 | 238.625133 | 3.957816 | 3.443438 | 3.476762 | 3.256384 | 3.210465 | 3.25313 | 3.240394 | 4.086007 | 3.347207 |
|  ORB_CUDA | ORB | 31.353487 | 250.58047 | 7.37172 | 7.92185 | 6.78292 | 6.6911 | 7.02198 | 6.32431 | 6.57911 | 6.63003 | 7.63138 |
|  ORB_CUDA | ORB_CUDA | 30.577485 | 268.56345 | 4.24 | 4.08908 | 4.16605 | 4.05041 | 4.18579 | 4.07111 | 4.17841 | 4.11226 | 4.11829 |
|  ORB_CUDA | FREAK | 52.82474 | 294.7652 | 26.05004 | 26.0329 | 26.17288 | 25.61477 | 25.78468 | 25.43468 | 25.3315 | 25.89088 | 27.16987 |
|  ORB_CUDA | AKAZE | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
|  AKAZE | BRISK | 229.33907 | 237.2436 | 228.2536 | 234.2653 | 230.9665 | 225.7471 | 229.7738 | 227.6581 | 226.9585 | 226.8153 | 225.7089 |
|  AKAZE | BRIEF | 54.609633 | 77.252663 | 51.531164 | 52.511223 | 50.454579 | 51.748617 | 56.216873 | 53.073919 | 50.979182 | 50.13447 | 52.19364 |
|  AKAZE | ORB | 55.737548 | 61.22377 | 55.35539 | 56.3615 | 55.60696 | 54.84333 | 54.69942 | 53.75804 | 53.55055 | 52.02341 | 59.95311 |
|  AKAZE | ORB_CUDA | 78.82535 | 335.0808 | 56.010733 | 56.279842 | 48.082229 | 48.186542 | 49.77843 | 49.450924 | 48.074538 | 48.52163 | 48.787832 |
|  AKAZE | FREAK | 76.23525 | 89.1829 | 75.4969 | 76.6507 | 76.8844 | 72.9002 | 77.2431 | 78.068 | 73.3685 | 71.3353 | 71.2225 |
|  AKAZE | AKAZE | 99.314 | 122.2455 | 95.6887 | 97.677 | 94.723 | 99.9132 | 98.7673 | 95.0662 | 100.1007 | 92.468 | 96.4904 |
|  AKAZE | SIFT | 68.1304 | 73.2069 | 65.8579 | 67.5421 | 66.1069 | 67.0908 | 66.5417 | 69.5929 | 67.2194 | 70.6174 | 67.528 |
|  SIFT | BRISK | 238.14079 | 262.5431 | 250.0639 | 233.6774 | 232.4148 | 234.4901 | 231.8929 | 235.0988 | 232.3026 | 231.4214 | 237.5029 |
|  SIFT | BRIEF | 76.0945552 | 100.655129 | 77.973121 | 75.042882 | 74.47681 | 76.51124 | 73.17165 | 73.641757 | 69.578527 | 69.568027 | 70.326409 |
|  SIFT | ORB | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |