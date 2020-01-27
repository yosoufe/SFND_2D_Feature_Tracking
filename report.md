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
- [Conclusion](#Conclusion)

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
for different detector for all 10 images and sorted based on the average column 
in decreasing order.

|  **Detector** | **Average** | **Image 0** | **Image 1** | **Image 2** | **Image 3** | **Image 4** | **Image 5** | **Image 6** | **Image 7** | **Image 8** | **Image 9** |  |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | --- |
|  **BRISK** | 276.3 | 264 | 282 | 282 | 277 | 297 | 279 | 289 | 272 | 267 | 254 |  |
|  **AKAZE** | 167 | 166 | 157 | 161 | 155 | 163 | 164 | 173 | 175 | 177 | 179 |  |
|  **FAST** | 149.1 | 149 | 152 | 150 | 155 | 149 | 149 | 156 | 150 | 138 | 143 |  |
|  **SIFT** | 138.6 | 138 | 132 | 124 | 137 | 134 | 140 | 137 | 148 | 159 | 137 |  |
|  **SHITOMASI** | 117.9 | 125 | 118 | 123 | 120 | 120 | 113 | 114 | 123 | 111 | 112 |  |
|  **ORB_CUDA** | 117.8 | 95 | 107 | 109 | 121 | 103 | 126 | 126 | 128 | 133 | 130 |  |
|  **ORB** | 116.1 | 92 | 102 | 106 | 113 | 109 | 125 | 130 | 129 | 127 | 128 |  |
|  **FAST_CUDA** | 110.3 | 104 | 107 | 110 | 113 | 109 | 115 | 113 | 121 | 95 | 116 |  |
|  **HARRIS** | 24.8 | 17 | 14 | 18 | 21 | 26 | 43 | 18 | 31 | 26 | 34 |  |

<a name="MP8" />

### MP8 - Performance Evaluation 2

The following table **counts the number of matched keypoints** for all 10 images using 
all possible combinations of detectors and descriptors. 
It is sorted based on the Average column in decreasing order. 
In the matching step, the BF approach is used with the descriptor distance ratio set to 0.8.

|  **Detector** | **Descriptor** | **Average** | **Images 0-1** | **Images 1-2** | **Images 2-3** | **Images 3-4** | **Images 4-5** | **Images 5-6** | **Images 6-7** | **Images 7-8** | **Images 8-9** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  **BRISK** | **BRIEF** | 189.3333333 | 178 | 205 | 185 | 180 | 182 | 196 | 206 | 189 | 183 |
|  **BRISK** | **SIFT** | 183.1111111 | 182 | 193 | 170 | 183 | 172 | 195 | 194 | 176 | 183 |
|  **BRISK** | **BRISK** | 174.4444444 | 171 | 176 | 157 | 176 | 174 | 187 | 173 | 172 | 184 |
|  **BRISK** | **FREAK** | 169.3333333 | 160 | 177 | 155 | 173 | 163 | 182 | 169 | 178 | 167 |
|  **BRISK** | **ORB** | 168.4444444 | 162 | 175 | 158 | 167 | 160 | 183 | 168 | 170 | 173 |
|  **AKAZE** | **SIFT** | 141.1111111 | 134 | 134 | 130 | 136 | 137 | 147 | 147 | 154 | 151 |
|  **BRISK** | **ORB_CUDA** | 140.7777778 | 138 | 152 | 129 | 138 | 118 | 155 | 140 | 148 | 149 |
|  **AKAZE** | **BRIEF** | 140.6666667 | 141 | 134 | 131 | 130 | 134 | 146 | 150 | 148 | 152 |
|  **AKAZE** | **AKAZE** | 139.8888889 | 138 | 138 | 133 | 127 | 129 | 146 | 147 | 151 | 150 |
|  **AKAZE** | **BRISK** | 135 | 137 | 125 | 129 | 129 | 131 | 132 | 142 | 146 | 144 |
|  **AKAZE** | **FREAK** | 131.8888889 | 126 | 129 | 127 | 121 | 122 | 133 | 144 | 147 | 138 |
|  **AKAZE** | **ORB** | 131.3333333 | 131 | 129 | 127 | 117 | 130 | 131 | 137 | 135 | 145 |
|  **FAST** | **BRIEF** | 122.1111111 | 119 | 130 | 118 | 126 | 108 | 123 | 131 | 125 | 119 |
|  **FAST** | **ORB** | 119 | 118 | 123 | 112 | 126 | 106 | 122 | 122 | 123 | 119 |
|  **AKAZE** | **ORB_CUDA** | 118.8888889 | 122 | 118 | 110 | 108 | 111 | 120 | 126 | 126 | 129 |
|  **FAST** | **SIFT** | 116.2222222 | 118 | 123 | 110 | 119 | 114 | 119 | 123 | 117 | 103 |
|  **FAST** | **ORB_CUDA** | 112.2222222 | 112 | 114 | 106 | 118 | 97 | 117 | 117 | 118 | 111 |
|  **SHITOMASI** | **BRIEF** | 104.8888889 | 115 | 111 | 104 | 101 | 102 | 102 | 100 | 109 | 100 |
|  **SHITOMASI** | **SIFT** | 103 | 112 | 109 | 104 | 103 | 99 | 101 | 96 | 106 | 97 |
|  **SHITOMASI** | **ORB** | 100.8888889 | 106 | 102 | 99 | 102 | 103 | 97 | 98 | 104 | 97 |
|  **FAST** | **BRISK** | 99.88888889 | 97 | 104 | 101 | 98 | 85 | 107 | 107 | 100 | 100 |
|  **FAST** | **FREAK** | 97.55555556 | 98 | 99 | 91 | 98 | 85 | 99 | 102 | 101 | 105 |
|  **SHITOMASI** | **ORB_CUDA** | 91.11111111 | 100 | 92 | 85 | 94 | 90 | 89 | 89 | 93 | 88 |
|  **SIFT** | **SIFT** | 88.88888889 | 82 | 81 | 85 | 93 | 90 | 81 | 82 | 102 | 104 |
|  **ORB_CUDA** | **SIFT** | 86.22222222 | 69 | 77 | 73 | 85 | 78 | 100 | 95 | 97 | 102 |
|  **SHITOMASI** | **FREAK** | 85.33333333 | 86 | 90 | 86 | 88 | 86 | 80 | 81 | 86 | 85 |
|  **SHITOMASI** | **BRISK** | 85.22222222 | 95 | 88 | 80 | 90 | 82 | 79 | 85 | 86 | 82 |
|  **ORB** | **ORB** | 84.77777778 | 67 | 70 | 72 | 84 | 91 | 101 | 92 | 93 | 93 |
|  **ORB** | **SIFT** | 84.77777778 | 67 | 79 | 78 | 79 | 82 | 95 | 95 | 94 | 94 |
|  **ORB_CUDA** | **BRISK** | 84.77777778 | 70 | 79 | 81 | 87 | 80 | 97 | 88 | 87 | 94 |
|  **ORB_CUDA** | **ORB** | 84.55555556 | 72 | 83 | 73 | 86 | 77 | 98 | 79 | 91 | 102 |
|  **ORB** | **BRISK** | 83.44444444 | 73 | 74 | 79 | 85 | 79 | 92 | 90 | 88 | 91 |
|  **ORB_CUDA** | **ORB_CUDA** | 79.33333333 | 67 | 81 | 67 | 77 | 75 | 89 | 76 | 86 | 96 |
|  **SIFT** | **BRIEF** | 78 | 86 | 78 | 76 | 85 | 69 | 74 | 76 | 70 | 88 |
|  **ORB** | **ORB_CUDA** | 77.88888889 | 66 | 71 | 71 | 73 | 78 | 86 | 85 | 80 | 91 |
|  **FAST_CUDA** | **SIFT** | 74.77777778 | 72 | 71 | 71 | 76 | 75 | 89 | 72 | 77 | 70 |
|  **FAST_CUDA** | **ORB** | 74.22222222 | 71 | 76 | 62 | 61 | 59 | 87 | 90 | 91 | 71 |
|  **FAST_CUDA** | **BRIEF** | 73.66666667 | 71 | 68 | 71 | 68 | 66 | 84 | 80 | 79 | 76 |
|  **FAST_CUDA** | **ORB_CUDA** | 73.22222222 | 59 | 64 | 75 | 76 | 60 | 83 | 85 | 85 | 72 |
|  **SIFT** | **FREAK** | 65.88888889 | 65 | 72 | 64 | 66 | 59 | 59 | 64 | 65 | 79 |
|  **SIFT** | **BRISK** | 65.77777778 | 64 | 66 | 62 | 66 | 59 | 64 | 64 | 67 | 80 |
|  **FAST_CUDA** | **FREAK** | 63.77777778 | 66 | 67 | 63 | 66 | 52 | 68 | 70 | 61 | 61 |
|  **ORB_CUDA** | **BRIEF** | 61.22222222 | 49 | 51 | 48 | 61 | 54 | 82 | 73 | 68 | 65 |
|  **ORB** | **BRIEF** | 60.55555556 | 49 | 43 | 45 | 59 | 53 | 78 | 68 | 84 | 66 |
|  **FAST_CUDA** | **BRISK** | 58 | 52 | 61 | 63 | 48 | 48 | 64 | 70 | 61 | 55 |
|  **ORB_CUDA** | **FREAK** | 47.77777778 | 44 | 39 | 50 | 50 | 42 | 49 | 47 | 50 | 59 |
|  **ORB** | **FREAK** | 46.66666667 | 42 | 36 | 44 | 47 | 44 | 51 | 52 | 48 | 56 |
|  **HARRIS** | **BRIEF** | 19.22222222 | 14 | 11 | 15 | 20 | 24 | 26 | 16 | 24 | 23 |
|  **HARRIS** | **SIFT** | 18.11111111 | 14 | 11 | 16 | 19 | 22 | 22 | 13 | 24 | 22 |
|  **HARRIS** | **ORB** | 18 | 12 | 12 | 15 | 18 | 24 | 20 | 15 | 24 | 22 |
|  **HARRIS** | **ORB_CUDA** | 17.55555556 | 12 | 13 | 16 | 18 | 23 | 18 | 13 | 24 | 21 |
|  **HARRIS** | **FREAK** | 16 | 13 | 13 | 15 | 15 | 17 | 20 | 12 | 21 | 18 |
|  **HARRIS** | **BRISK** | 15.77777778 | 12 | 10 | 14 | 15 | 16 | 16 | 15 | 23 | 21 |
|  **SHITOMASI** | **AKAZE** | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
|  **HARRIS** | **AKAZE** | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
|  **FAST** | **AKAZE** | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
|  **FAST_CUDA** | **AKAZE** | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
|  **BRISK** | **AKAZE** | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
|  **ORB** | **AKAZE** | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
|  **ORB_CUDA** | **AKAZE** | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
|  **SIFT** | **ORB** | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
|  **SIFT** | **ORB_CUDA** | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
|  **SIFT** | **AKAZE** | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

<a name="MP9" />

### MP9 - Performance Evaluation 3

The following table describes **the time it takes for keypoint detection and descriptor extraction in milliseconds**.
The table is sorted based on the Average column and in increasing order. In this table the time for images 1 to 9 are 
only used to calculate the average time. Because the first image took a lot of time probably for initialization of 
objects which should be ignored for long stream of sensor frame data.

The measurement for the CUDA ones is not 100% fair because it is not optimized well. There are a lot copy and paste between 
the CPU memory and GPU memory which is happening and can be optimized later. Though they are showing good performance.

|  **Detector** | **Descriptor** | **Average** | **Image 0** | **Image 1** | **Image 2** | **Image 3** | **Image 4** | **Image 5** | **Image 6** | **Image 7** | **Image 8** | **Image 9** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  **FAST** | **BRIEF** | 1.053317111 | 2.99997 | 1.582577 | 1.358646 | 0.963147 | 0.92651 | 0.936316 | 0.94373 | 0.940347 | 0.906206 | 0.922375 |
|  **FAST_CUDA** | **BRIEF** | 1.109157222 | 157.998243 | 1.468696 | 1.199434 | 1.109147 | 0.946704 | 1.251766 | 0.982208 | 0.853709 | 1.322367 | 0.848384 |
|  **FAST** | **ORB** | 1.299221889 | 2.42679 | 1.565492 | 1.326094 | 1.325924 | 1.206796 | 1.213173 | 1.211427 | 1.426874 | 1.208152 | 1.209065 |
|  **FAST_CUDA** | **ORB** | 1.346772222 | 146.962823 | 1.46576 | 1.36623 | 1.386564 | 1.274144 | 1.454577 | 1.263938 | 1.169716 | 1.586389 | 1.153632 |
|  **FAST** | **ORB_CUDA** | 1.445710222 | 212.438476 | 1.47742 | 1.189741 | 1.137179 | 1.421786 | 1.137039 | 1.158919 | 1.389528 | 1.181505 | 2.918275 |
|  **FAST_CUDA** | **ORB_CUDA** | 1.745415889 | 273.7229 | 1.607643 | 1.194235 | 3.625495 | 1.908669 | 1.093889 | 2.07254 | 1.24986 | 1.154095 | 1.802317 |
|  **ORB_CUDA** | **BRIEF** | 3.745947 | 271.137181 | 5.096232 | 3.557807 | 4.07654 | 3.291038 | 3.391783 | 3.323876 | 3.317993 | 4.280619 | 3.377635 |
|  **ORB_CUDA** | **ORB_CUDA** | 4.712475556 | 348.43884 | 4.68586 | 4.12807 | 4.08778 | 4.0585 | 5.19554 | 4.13421 | 6.82444 | 4.1368 | 5.16108 |
|  **ORB** | **BRIEF** | 5.885073111 | 179.645587 | 6.636411 | 5.943649 | 5.869517 | 6.113862 | 5.598134 | 5.661711 | 5.535918 | 5.691472 | 5.914984 |
|  **ORB** | **ORB_CUDA** | 7.152456667 | 354.486 | 8.12834 | 9.06167 | 6.57548 | 6.92407 | 6.46922 | 6.41625 | 6.72828 | 7.31616 | 6.75264 |
|  **ORB_CUDA** | **ORB** | 7.849332222 | 232.60944 | 7.44543 | 8.79455 | 7.0757 | 7.21621 | 8.55718 | 10.24726 | 7.29126 | 6.62788 | 7.38852 |
|  **SHITOMASI** | **ORB_CUDA** | 8.350436667 | 233.2469 | 10.939327 | 8.117964 | 8.12404 | 8.211285 | 7.799025 | 7.964039 | 7.958582 | 8.168267 | 7.871401 |
|  **ORB** | **ORB** | 8.970226667 | 172.24717 | 9.69145 | 8.91861 | 9.04568 | 8.92136 | 8.56142 | 8.97815 | 9.03763 | 8.93639 | 8.64135 |
|  **HARRIS** | **ORB_CUDA** | 9.964748444 | 286.3153 | 11.385028 | 8.279192 | 8.322965 | 8.913753 | 15.831041 | 8.065976 | 9.232135 | 8.520355 | 11.132291 |
|  **SHITOMASI** | **BRIEF** | 10.95350622 | 13.923739 | 11.431123 | 11.252161 | 10.954395 | 11.011525 | 10.679642 | 10.893686 | 10.524368 | 10.892559 | 10.942097 |
|  **SHITOMASI** | **ORB** | 11.187032 | 12.899116 | 11.688254 | 11.231835 | 10.994681 | 11.220318 | 11.002009 | 11.065403 | 11.179248 | 11.068184 | 11.233356 |
|  **HARRIS** | **BRIEF** | 12.30653644 | 12.695212 | 11.009819 | 11.075056 | 11.096764 | 11.196599 | 18.193727 | 10.826252 | 11.8753 | 11.401511 | 14.0838 |
|  **HARRIS** | **ORB** | 12.914753 | 13.834563 | 11.668586 | 11.790947 | 11.559689 | 11.890136 | 18.752983 | 11.474905 | 12.661787 | 11.832869 | 14.600875 |
|  **FAST_CUDA** | **SIFT** | 15.67531322 | 177.0936 | 31.59223 | 14.63578 | 13.346689 | 13.42385 | 13.776767 | 13.959759 | 13.574945 | 12.825784 | 13.942015 |
|  **FAST** | **SIFT** | 18.55661644 | 20.44566 | 24.89168 | 23.69181 | 18.267594 | 17.495878 | 17.469956 | 17.150448 | 15.784121 | 16.760906 | 15.497155 |
|  **FAST** | **FREAK** | 23.91857356 | 29.53406 | 26.674885 | 23.604912 | 23.241422 | 23.740264 | 24.250991 | 23.29881 | 23.774303 | 23.462837 | 23.218738 |
|  **SHITOMASI** | **SIFT** | 25.15995111 | 37.0546 | 28.9663 | 25.2991 | 25.4447 | 25.9747 | 24.79497 | 24.7742 | 23.9469 | 23.27439 | 23.9643 |
|  **ORB_CUDA** | **FREAK** | 26.44493778 | 305.8222 | 26.18751 | 26.0027 | 26.11336 | 26.02772 | 26.43021 | 26.9688 | 27.69978 | 26.34772 | 26.22664 |
|  **FAST_CUDA** | **FREAK** | 27.145263 | 227.8184 | 24.785551 | 24.315477 | 24.230521 | 23.954658 | 24.014564 | 48.86451 | 24.89119 | 25.27318 | 23.977716 |
|  **HARRIS** | **SIFT** | 27.23954444 | 27.8944 | 25.4521 | 26.7728 | 26.3623 | 26.3286 | 34.999 | 25.4278 | 25.7563 | 25.7298 | 28.3272 |
|  **ORB** | **FREAK** | 28.50071667 | 180.4342 | 29.26983 | 28.06853 | 27.84285 | 28.17492 | 28.51226 | 28.65839 | 28.52569 | 28.76401 | 28.68997 |
|  **ORB_CUDA** | **SIFT** | 28.99877778 | 292.8145 | 30.71559 | 29.03035 | 30.51468 | 26.86088 | 27.94057 | 28.54054 | 29.56323 | 29.49618 | 28.32698 |
|  **SHITOMASI** | **FREAK** | 30.11050667 | 39.7315 | 32.61284 | 29.96465 | 29.78623 | 29.9631 | 29.95743 | 29.71551 | 29.90905 | 29.42337 | 29.66238 |
|  **ORB** | **SIFT** | 31.90811778 | 169.0492 | 30.97777 | 31.65272 | 32.46393 | 32.00061 | 31.18449 | 30.29155 | 31.89958 | 33.2096 | 33.49281 |
|  **HARRIS** | **FREAK** | 32.31666333 | 39.0161 | 32.8216 | 30.04267 | 31.00927 | 30.88008 | 38.6214 | 30.24876 | 31.40366 | 31.71443 | 34.1081 |
|  **AKAZE** | **ORB_CUDA** | 48.94504678 | 290.9393 | 56.744287 | 48.598898 | 48.073361 | 47.065489 | 46.861346 | 47.18743 | 47.76338 | 48.0716 | 50.13963 |
|  **AKAZE** | **BRIEF** | 52.90521433 | 75.22233 | 52.658121 | 49.533982 | 53.023238 | 52.401023 | 51.954543 | 51.438697 | 57.320101 | 53.939365 | 53.877859 |
|  **AKAZE** | **ORB** | 56.59247889 | 64.1881 | 57.12471 | 58.08617 | 54.30606 | 55.60572 | 63.19322 | 55.91102 | 56.1354 | 54.94627 | 54.02374 |
|  **AKAZE** | **SIFT** | 68.65421111 | 102.5976 | 69.635 | 67.9778 | 69.8721 | 72.9825 | 68.2255 | 68.7166 | 65.7662 | 66.3845 | 68.3277 |
|  **SIFT** | **BRIEF** | 74.81155667 | 82.614149 | 77.035598 | 74.966386 | 77.886723 | 76.841677 | 75.466761 | 71.941593 | 74.719895 | 71.955934 | 72.489443 |
|  **AKAZE** | **FREAK** | 76.94788889 | 105.3684 | 80.8501 | 75.5262 | 78.3936 | 75.0875 | 76.4311 | 74.7134 | 76.5855 | 76.9907 | 77.9529 |
|  **AKAZE** | **AKAZE** | 100.1262111 | 114.798 | 97.7026 | 97.3977 | 96.0372 | 100.7127 | 103.331 | 101.5946 | 102.8714 | 97.9008 | 103.5879 |
|  **SIFT** | **FREAK** | 102.1235 | 123.6355 | 103.8898 | 98.7855 | 100.8077 | 101.1288 | 101.1354 | 120.399 | 99.0354 | 97.3229 | 96.607 |
|  **SIFT** | **SIFT** | 112.4137444 | 177.294 | 127.1477 | 112.5496 | 114.0314 | 111.5887 | 112.6139 | 106.5636 | 110.8867 | 107.1347 | 109.2074 |
|  **FAST** | **BRISK** | 179.792517 | 181.619687 | 183.453896 | 179.669443 | 179.065139 | 178.334342 | 178.383331 | 179.242605 | 179.860928 | 180.547433 | 179.575536 |
|  **FAST_CUDA** | **BRISK** | 180.138267 | 321.43 | 180.426103 | 178.211679 | 178.99409 | 180.184217 | 178.295973 | 180.353697 | 183.02612 | 180.740198 | 181.012326 |
|  **ORB** | **BRISK** | 185.5985856 | 352.987 | 186.21864 | 186.03177 | 184.3654 | 186.6369 | 184.80953 | 185.31637 | 182.89314 | 186.01731 | 188.09821 |
|  **ORB_CUDA** | **BRISK** | 186.0417211 | 449.319 | 184.06984 | 186.15272 | 184.41314 | 204.58016 | 186.03915 | 182.73717 | 182.87286 | 184.07826 | 179.43219 |
|  **SHITOMASI** | **BRISK** | 188.9530489 | 196.0355 | 189.0179 | 188.0827 | 189.8585 | 188.3563 | 188.0533 | 190.3591 | 188.6219 | 188.69944 | 189.5283 |
|  **HARRIS** | **BRISK** | 189.8388667 | 190.0197 | 188.6451 | 190.1448 | 188.2337 | 188.3955 | 195.2791 | 187.6688 | 190.8162 | 187.799 | 191.5676 |
|  **BRISK** | **ORB_CUDA** | 212.9649622 | 454.99 | 214.05448 | 212.82094 | 214.03471 | 212.51219 | 211.89885 | 212.93824 | 211.65164 | 212.40718 | 214.36643 |
|  **BRISK** | **BRIEF** | 213.6099872 | 230.95017 | 212.757753 | 211.315908 | 222.575929 | 215.181463 | 214.188034 | 214.038365 | 208.841128 | 210.097831 | 213.493474 |
|  **BRISK** | **ORB** | 215.1179756 | 217.5021 | 214.12856 | 214.43588 | 213.68648 | 214.78837 | 213.30666 | 216.07682 | 215.17843 | 217.00713 | 217.45345 |
|  **BRISK** | **FREAK** | 234.9841 | 242.6038 | 233.286 | 230.953 | 237.3906 | 236.5473 | 232.3055 | 233.5356 | 240.5366 | 234.9857 | 235.3166 |
|  **BRISK** | **SIFT** | 235.5753778 | 252.2609 | 252.2019 | 235.7736 | 232.3123 | 233.0225 | 234.6478 | 234.8517 | 232.2658 | 232.4564 | 232.6464 |
|  **AKAZE** | **BRISK** | 235.8537889 | 267.6838 | 236.0744 | 240.5033 | 236.6342 | 231.6402 | 235.172 | 234.1882 | 240.4104 | 237.1677 | 230.8937 |
|  **SIFT** | **BRISK** | 241.2832333 | 284.522 | 241.1451 | 250.5047 | 238.5781 | 243.5427 | 236.0226 | 242.2478 | 240.1142 | 238.5446 | 240.8493 |
|  **BRISK** | **BRISK** | 396.1786667 | 397.84 | 396.874 | 392.121 | 389.857 | 391.535 | 396.524 | 399.865 | 399.629 | 399.223 | 399.98 |
|  **SHITOMASI** | **AKAZE** | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
|  **HARRIS** | **AKAZE** | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
|  **FAST** | **AKAZE** | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
|  **FAST_CUDA** | **AKAZE** | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
|  **BRISK** | **AKAZE** | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
|  **ORB** | **AKAZE** | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
|  **ORB_CUDA** | **AKAZE** | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
|  **SIFT** | **ORB** | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
|  **SIFT** | **ORB_CUDA** | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
|  **SIFT** | **AKAZE** | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |


<a name="Conclusion" />

## Conclusion

- In the number of detected keypoints, the three best methods are BRISK, AKAZE and FAST with 276.3, 167 and 149.1 keypoints in average.
- In number of matched keypoints, the three best detector-descriptor matches are BRISK-BRIEF, BRISK-SIFT and BRISk-BRISK and that can be because BRISK has the most number of keypoints in general.
- In time required for detection and matching, the three best detector-descriptor matches are FAST-BRIEF, FAST_CUDA-BRIEF and FAST-ORB.

The above results were not stable. They may change the order in different runs.

It is difficult ot judge which setup is better because we do not measure the correctness of matched keypoints in these measurements. And also as it can be seen, the matches that had fewer number of keypoints usually they got a better timing 
results which is reasonable. Therefor I believe we should consider the trade off between number of keypoints and 
the time it takes to match the keypoints.

I decided to create a following table which is showing the average time it took for each keypoint and based on this table 
the three best detectors and matchers would be **FAST**, **FAST_CUDA** and **ORB_CUDA**.

|  Detector | Descriptor | Average Time / Average Number of Keypoints |
| --- | --- | --- |
|  FAST | BRIEF | 0.00862589081 |
|  FAST | ORB | 0.010917831 |
|  FAST | ORB_CUDA | 0.01288256634 |
|  FAST_CUDA | BRIEF | 0.01505643288 |
|  FAST_CUDA | ORB | 0.01814513473 |
|  FAST_CUDA | ORB_CUDA | 0.02383724279 |
|  ORB_CUDA | ORB_CUDA | 0.05940095238 |
|  ORB_CUDA | BRIEF | 0.06118606715 |
|  SHITOMASI | ORB_CUDA | 0.09165113415 |
|  ORB | ORB_CUDA | 0.0918289729 |
|  ORB_CUDA | ORB | 0.09283047306 |
|  ORB | BRIEF | 0.09718469358 |
|  SHITOMASI | BRIEF | 0.1044296144 |
|  ORB | ORB | 0.1058087025 |
|  SHITOMASI | ORB | 0.1108846784 |
|  FAST | SIFT | 0.1596649598 |
|  FAST_CUDA | SIFT | 0.2096252883 |
|  SHITOMASI | SIFT | 0.24427137 |
|  FAST | FREAK | 0.245179 |
|  ORB_CUDA | SIFT | 0.3363260309 |
|  SHITOMASI | FREAK | 0.3528575 |
|  AKAZE | BRIEF | 0.3761034194 |
|  ORB | SIFT | 0.3763736042 |
|  AKAZE | ORB_CUDA | 0.4116873093 |
|  FAST_CUDA | FREAK | 0.4256225906 |
|  AKAZE | ORB | 0.4309071997 |
|  AKAZE | SIFT | 0.4865259055 |
|  ORB_CUDA | FREAK | 0.5534986977 |
|  HARRIS | ORB_CUDA | 0.5676122532 |
|  AKAZE | FREAK | 0.5834296546 |
|  ORB | FREAK | 0.6107296429 |
|  HARRIS | BRIEF | 0.6402244393 |
|  AKAZE | AKAZE | 0.715755282 |
|  HARRIS | ORB | 0.7174862778 |
|  SIFT | BRIEF | 0.9591225214 |
|  BRISK | BRIEF | 1.128221763 |
|  SIFT | SIFT | 1.264654625 |
|  BRISK | ORB | 1.277085607 |
|  BRISK | SIFT | 1.286516019 |
|  BRISK | FREAK | 1.387701378 |
|  HARRIS | SIFT | 1.504023926 |
|  BRISK | ORB_CUDA | 1.512774002 |
|  SIFT | FREAK | 1.549935076 |
|  AKAZE | BRISK | 1.747065103 |
|  FAST | BRISK | 1.799925087 |
|  HARRIS | FREAK | 2.019791458 |
|  ORB_CUDA | BRISK | 2.19446329 |
|  SHITOMASI | BRISK | 2.217180495 |
|  ORB | BRISK | 2.224217403 |
|  BRISK | BRISK | 2.271087898 |
|  FAST_CUDA | BRISK | 3.10583219 |
|  SIFT | BRISK | 3.668157264 |
|  HARRIS | BRISK | 12.03204085 |
|  SHITOMASI | AKAZE | N/A |
|  HARRIS | AKAZE | N/A |
|  FAST | AKAZE | N/A |
|  FAST_CUDA | AKAZE | N/A |
|  BRISK | AKAZE | N/A |
|  ORB | AKAZE | N/A |
|  ORB_CUDA | AKAZE | N/A |
|  SIFT | ORB | N/A |
|  SIFT | ORB_CUDA | N/A |
|  SIFT | AKAZE | N/A |