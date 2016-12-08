#ifndef FACE_DETECTION_SAVE_H
#define FACE_DETECTION_SAVE_H

//C++ related includes.
#include <cstdio>
#include <cmath>
#include <sys/stat.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

// ROS related includes.
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/String.h>
#include <image_transport/image_transport.h>

// OpenCV related includes.
#include <cv_bridge/cv_bridge.h>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// Third party includes for tracking.
#include "../cf_libs/kcf/kcf_tracker.hpp"

// Debug defines.
// Include this if you want to have visual output.
#define DEBUG


using namespace cv;


/**
 * @brief      Class for face detection and tracking.
 */
class FaceDetectionSave
{
public:

    /**
     * @brief      Constructor for the class.
     */
    FaceDetectionSave();

    /**
     * @brief      Destructor.
     */
    ~FaceDetectionSave();

    /**
     * @brief      Function for detecting and displaying the faces.
     *
     * @param[in]  frame  The frame
     */
    void detectAndDisplay(cv::Mat frame);

    /**
     * @brief      Track the object.
     */
    void track();

private:
    // Global variables.

    int i;

    // The ros node handle.
    ros::NodeHandle m_node;

    std::string m_windowName{"Face detector"};
    std::string m_directory; //{"/work/pangerca/catkin_ws/src/face_detection_tracker/"};
    std::string output_folder; //{"/work/pangerca/faces_ics/tmp/"};
    std::string m_camera_topic;

    // Buffer for publishers, subscibers.
    int m_queuesize = 2;

    ////////////////////////////
    /// Face detection part. ///
    ////////////////////////////

    // Helper member variables for image transformation.
    image_transport::ImageTransport m_it;
    image_transport::Subscriber m_imageSub;

    // POinter to the cv image.
    cv_bridge::CvImagePtr m_cvPtr;

    // Name the haarcascades for frontal and profile face detection.
    std::string m_frontalFaceCascadeName{"haarcascade_frontalface_alt.xml"};
    std::string m_profilefaceCascadeName{"haarcascade_profileface.xml"};

    // Cascade classifiers.
    cv::CascadeClassifier m_frontalfaceCascade;
    cv::CascadeClassifier m_profilefaceCascade;

    /**
     * @brief      Callback for the sensor_msgs::Image.
     *
     * @param[in]  msg   The image in a form of a sensor_msgs::Image.
     */
    void imageCallback(const sensor_msgs::ImageConstPtr &msg);

    // Point in the upper left corner.
    cv::Point m_p1;

    // Point in the lower right corner.
    cv::Point m_p2;

    // Height and width of the bounding box.
    int m_width;
    int m_height;

    // detecting frontal or profile face
    int faceMethod;
    // we are skiping frames for detection, cpu-s are less overloaded
    #define SKIP_FRAMES 10
    int skipFrames;

    //Size of face frame in pixels, dim x = dim y
    #define FR_SI 100

    // frame which hold the face
    cv::Mat face_for_save;

    Mat norm_0_255(InputArray _src);
};

#endif // FACE_DETECTION_SAVE_H
