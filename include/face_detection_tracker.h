#ifndef FACE_DETECTION_TRACKER_H
#define FACE_DETECTION_TRACKER_H


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
#include <perception_msgs/Rect.h>

// Self defined includes.
#include <perception_msgs/Rect.h>


// Debug defines.
// Include this if you want to have visual output.
#define DEBUG


using namespace cv;


/**
 * @brief      Class for face detection and tracking.
 */
class FaceDetectionTracker
{
public:

    /**
     * @brief      Constructor for the class.
     */
    FaceDetectionTracker();

    /**
     * @brief      Destructor.
     */
    ~FaceDetectionTracker();

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
    static bool m_newImage_static;
    static bool m_newBB_static;

    // The ros node handle.
    ros::NodeHandle m_node;

    std::string m_windowName{"Face detector"};
    std::string m_windowName0{"Tracked object"};
    std::string m_directory{"/work/pangerca/catkin_ws/src/face_detection_tracker/"};

    // Buffer for publishers, subscibers.
    int m_queuesize = 2;

    ////////////////////////////
    /// Face detection part. ///
    ////////////////////////////

    // Helper member variables for image transformation.
    image_transport::ImageTransport m_it;
    image_transport::Subscriber m_imageSub;
    image_transport::Publisher m_imagePub;

    // Publisher.
    ros::Publisher m_perceptPub;

    // POinter to the cv image.
    cv_bridge::CvImagePtr m_cvPtr;

    // Name the haarcascades for frontal and profile face detection.
    std::string m_frontalFaceCascadeName{"haarcascade_frontalface_alt.xml"};
    std::string m_profilefaceCascadeName{"haarcascade_profileface.xml"};

    // Cascade classifiers.
    cv::CascadeClassifier m_frontalfaceCascade;
    cv::CascadeClassifier m_profilefaceCascade;

    // Variable to hold the detection rectangle.
    perception_msgs::Rect m_msgRect;

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

    //////////////////////
    /// Tracking part. ///
    //////////////////////

    // Cv Bridge variables for transforming sensor_msgs::Image into cv::Mat
    cv_bridge::CvImagePtr m_inImg;

    perception_msgs::Rect m_inBb;
    perception_msgs::Rect m_outBb;

    // local variables
    cv::Mat img;
    cv::Rect bb;

    // Detected face number to track.
    int i = 0;

    // Declare and initialize subscribers, rgb image and the 2D region to track.
    ros::Subscriber rgbimgSub;

    // Face detection subscriber.
    ros::Subscriber bbSub;

    //Declare and initialize publishers, 2D region tracked region.
    ros::Publisher bbPub;

    // Tracker parameters.
    cf_tracking::KcfParameters m_paras;

    std::vector<cf_tracking::KcfTracker*> vKCF;

    // Declare tracker.
    cf_tracking::KcfTracker *cKCF;

    // The tracker is running.
    bool tracking = false;

    // If the tracker is on frame.
    bool targetOnFrame = false;

};

#endif // FACE_DETECTION_TRACKER_H
