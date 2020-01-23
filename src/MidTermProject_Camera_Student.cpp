/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"

#include "argparse.h"

using namespace std;

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{
    // parse command line arguments:
    static const char *const usage[] = {
        "./2D_feature_tracking [args]\n"
        "For example: ./2D_feature_tracking --detector_type=BRISK --matcher_type=MAT_FLANN --descriptor_type=DES_BINARY --selector_type=SEL_KNN",
        NULL,
        NULL
    };

    // default values for arguments
    const char* detectorTypeC = "SHITOMASI";      // SHITOMASI, HARRIS, FAST, BRISK, ORB, AKAZE, and SIFT
    const char* matcherTypeC = "MAT_BF";          // MAT_BF, MAT_FLANN
    const char* descriptorTypeC = "BRISK";        // BRISK BRIEF, ORB, FREAK, AKAZE, SIFT
    const char* selectorTypeC = "SEL_NN";         // SEL_NN, SEL_KNN
    bool bFocusOnVehicle = false;
    bool bLimitKpts = false;
    bool bQuiet = false;

    struct argparse_option options[] = {
        OPT_HELP(),
        OPT_GROUP("Optional Arguments: "),
        OPT_STRING('\0', "detector_type", &detectorTypeC, "detector type, options: SHITOMASI, HARRIS, FAST, BRISK, ORB, AKAZE, and SIFT"
                                                          "\n\t\t\t\tif compiled (WITH_CUDA on): ORB_CUDA, FAST_CUDA"
                                                          "\n\t\t\t\tdefault: SHITOMASI"),
        OPT_STRING('\0', "matcher_type", &matcherTypeC, "matcher type, options: MAT_BF, MAT_FLANN,"
                                                        "\n\t\t\t\tif compiled (WITH_CUDA on): MAT_BF_CUDA"
                                                        "\n\t\t\t\tdefault: MAT_BF"),
        OPT_STRING('\0', "descriptor_type", &descriptorTypeC, "descriptor type, options: BRISK BRIEF, ORB, FREAK, AKAZE, SIFT"
                                                              "\n\t\t\t\tif compiled (WITH_CUDA on): ORB_CUDA"
                                                              "\n\t\t\t\tdefault: BRISK"),
        OPT_STRING('\0', "selector_type", &selectorTypeC, "selector type, options: SEL_NN, SEL_KNN"
                                                          "\n\t\t\t\tdefault: SEL_NN"),
        OPT_BOOLEAN('f', "focus_on_vehicle", &bFocusOnVehicle, "To focus on only keypoints that are on the preceding vehicle."),
        OPT_BOOLEAN('l', "limit_keypoints", &bLimitKpts, "To limit the number of keypoints to maximum 50 keypoints."),
        OPT_BOOLEAN('q', "quiet", &bQuiet, "If this flaged is chosen no image would be shown. Good for performance measurement"),
        OPT_END(),
    };
    struct argparse argparse;
    argparse_init(&argparse, options, usage, 0);
    argparse_describe(&argparse, "\nExplores differenet 2d keypoint detector, descriptor and matching", NULL);
    argc = argparse_parse(&argparse, argc, argv);

    std::string detectorType(detectorTypeC);
    std::string matcherType(matcherTypeC);    
    std::string descriptorType(descriptorTypeC);
    std::string selectorType(selectorTypeC);

    /* INIT VARIABLES AND DATA STRUCTURES */

    // data location
    string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // misc
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
    bool bVis = false;            // visualize results

    /* MAIN LOOP OVER ALL IMAGES */

    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
    {
        /* LOAD IMAGE INTO BUFFER */

        // assemble filenames for current index
        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file and convert to grayscale
        cv::Mat img, imgGray;
        img = cv::imread(imgFullFilename);
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

        //// STUDENT ASSIGNMENT
        //// TASK MP.1 -> replace the following code with ring buffer of size dataBufferSize

        // push image into data frame buffer
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
        

        //// EOF STUDENT ASSIGNMENT
        if (!bQuiet) cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;

        /* DETECT IMAGE KEYPOINTS */

        // extract 2D keypoints from current image
        vector<cv::KeyPoint> keypoints; // create empty feature list for current image

        //// STUDENT ASSIGNMENT
        //// TASK MP.2 -> add the following keypoint detectors in file matching2D.cpp and enable string-based selection based on detectorType
        //// -> HARRIS, FAST, BRISK, ORB, AKAZE, SIFT

        if (detectorType.compare("SHITOMASI") == 0)
        {
            detKeypointsShiTomasi(keypoints, imgGray, false);
        }
        else if (detectorType.compare("HARRIS") == 0)
        {
            detKeypointsHarris(keypoints, imgGray, false);
        }
        else if (detectorType.compare("FAST") == 0 || 
                    detectorType.compare("BRISK") == 0 || 
                    detectorType.compare("ORB") == 0 || 
                    detectorType.compare("AKAZE") == 0  || 
                    detectorType.compare("SIFT") == 0 || 
                    detectorType.compare("ORB_CUDA") == 0 || 
                    detectorType.compare("FAST_CUDA") == 0 )
        {
            detKeypointsModern(keypoints, imgGray, detectorType, false);
        }
        //// EOF STUDENT ASSIGNMENT

        //// STUDENT ASSIGNMENT
        //// TASK MP.3 -> only keep keypoints on the preceding vehicle

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

        //// EOF STUDENT ASSIGNMENT

        // optional : limit number of keypoints (helpful for debugging and learning)
        if (bLimitKpts)
        {
            int maxKeypoints = 50;

            if (detectorType.compare("SHITOMASI") == 0)
            { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
            }
            cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
            if (!bQuiet) cout << " NOTE: Keypoints have been limited!" << endl;
        }

        // push keypoints and descriptor for current frame to end of data buffer
        (dataBuffer.end() - 1)->keypoints = keypoints;
        if (!bQuiet) cout << "#2 : DETECT KEYPOINTS done" << endl;

        /* EXTRACT KEYPOINT DESCRIPTORS */

        //// STUDENT ASSIGNMENT
        //// TASK MP.4 -> add the following descriptors in file matching2D.cpp and enable string-based selection based on descriptorType
        //// -> BRIEF, ORB, FREAK, AKAZE, SIFT

        cv::Mat descriptors;
        descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType);
        //// EOF STUDENT ASSIGNMENT

        // push descriptors for current frame to end of data buffer
        (dataBuffer.end() - 1)->descriptors = descriptors;

        if (!bQuiet) cout << "#3 : EXTRACT DESCRIPTORS done" << endl;

        if (dataBuffer.size() > 1) // wait until at least two images have been processed
        {

            /* MATCH KEYPOINT DESCRIPTORS */

            vector<cv::DMatch> matches;

            //// STUDENT ASSIGNMENT
            //// TASK MP.5 -> add FLANN matching in file matching2D.cpp
            //// TASK MP.6 -> add KNN match selection and perform descriptor distance ratio filtering with t=0.8 in file matching2D.cpp

            matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                             (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                             matches, descriptorType, matcherType, selectorType);
            
            cout << "Number of Matched Keypoints: " <<  matches.size() << endl;

            //// EOF STUDENT ASSIGNMENT

            // store matches in current data frame
            (dataBuffer.end() - 1)->kptMatches = matches;

            if (!bQuiet) cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << endl;

            // visualize matches between current and previous image
            bVis = !bQuiet;
            if (bVis)
            {
                cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
                cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                                (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                                matches, matchImg,
                                cv::Scalar::all(-1), cv::Scalar::all(-1),
                                vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                string windowName = "Matching keypoints between two camera images";
                cv::namedWindow(windowName, 7);
                cv::imshow(windowName, matchImg);
                cout << "Press key to continue to next image" << endl;
                cv::waitKey(0); // wait for key to be pressed
            }
        }

    } // eof loop over all images

    return 0;
}
