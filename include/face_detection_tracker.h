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

//opening saved images on hard drive
#include<dirent.h>

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

    static bool m_newBB_static;

    // The ros node handle.
    ros::NodeHandle m_node;

    std::string m_windowName{"Face detector"};
    std::string m_windowName0{"Tracked object"};
    std::string m_directory; //{"/work/pangerca/catkin_ws/src/face_detection_tracker/"};

    // Buffer for publishers, subscibers.
    int m_queuesize = 2;

    ////////////////////////////
    /// Face detection part. ///
    ////////////////////////////

    // Helper member variables for image transformation.
    image_transport::ImageTransport m_it;
    image_transport::Subscriber m_imageSub;
    //image_transport::Publisher m_imagePub;

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
    // skiping frames at detection part -> less overloaded cpu-s
    int SKIP_FRAMES;
    // counter for frames
    int skipFramesCounter;

    //////////////////////
    /// Tracking part. ///
    //////////////////////

    // Cv Bridge variables for transforming sensor_msgs::Image into cv::Mat
    cv_bridge::CvImagePtr m_inImg;

    perception_msgs::Rect m_outBb;

    // local variables
    //cv::Mat img;
    cv::Rect bb;

    std::map<int, std::string> my_map;

    //Declare and initialize publishers, 2D region tracked region.
    ros::Publisher bbPub;

    // Tracker parameters.
    cf_tracking::KcfParameters m_paras;

    //std::vector<cf_tracking::KcfTracker*> vKCF;

    // Declare tracker.
    cf_tracking::KcfTracker *cKCF;

    // The tracker is running.
    bool tracking = false;

    // If the tracker is on frame.
    bool targetOnFrame = false;

    // for reseting a tracker, not implemented yet
    ros::Duration timeout;
    ros::Time start_time;

    // part for fisherfaces
    // These vectors hold the images and corresponding labels.
    vector<Mat> images;
    // new person, new number
    vector<int> labels;

    // history of people who were detected last LIST_SIZE times
    #define LIST_SIZE 10
    int history_ints[LIST_SIZE];
    int index_list;

    // variable is used for graphisc of the tracker
    int indexOfPerson;

    //number of people that are in date base.
    int numberOfPeople;

    // size of images
    cv::Point pic_size;

    // Eigenfaces model
    Ptr<FaceRecognizer> model;

    // normalize pixels to byte size
    Mat norm_0_255(InputArray _src);

    //reading database images from folder
    bool readImages(std::string person, int tag);

    // location of database images
    std::string dirName;
    // for reading images
    DIR *dir;
    struct dirent *ent;
};

#endif // FACE_DETECTION_TRACKER_H
