#include "face_detection_tracker.h"

// Initialize static members.
bool FaceDetectionTracker::m_newBB_static = false;

/**
 * @brief      Constructor for the class.
 */
FaceDetectionTracker::FaceDetectionTracker() :
        m_it(m_node)
{
    ///////////////////////
    /// Detection part. ///
    ///////////////////////

    // Subscribe to input video feed and publish output video feed.
    m_imageSub = m_it.subscribe("/pseye_camera/image_raw", 1, &FaceDetectionTracker::imageCallback, this);


    // Load the cascades.
    // // Frontal face.
    if(!m_frontalfaceCascade.load(m_directory + m_frontalFaceCascadeName))
    {
        ROS_ERROR("Error loading frontal face cascade!");
        return;
    }

    // Profile face.
    if(!m_profilefaceCascade.load(m_directory + m_profilefaceCascadeName))
    {
        ROS_ERROR("Error loading profile face cascade!");
        return;
    }


    /////////////////////
    /// Tracker part. ///
    /////////////////////

    bbPub = m_node.advertise<perception_msgs::Rect>("tracker/bb", m_queuesize);

    m_paras.enableTrackingLossDetection = true;
    // paras.psrThreshold = 10; // lower more flexible
    m_paras.psrThreshold = 13.5; // higher more restricted to changes
    m_paras.psrPeakDel = 2; // 1;
}

///////////////////////
/// Detection part. ///
///////////////////////

/**
 * @brief      Destructor.
 */
FaceDetectionTracker::~FaceDetectionTracker()
{}

/**
 * @brief      Callback for the sensor_msgs::Image.
 *
 * @param[in]  msg   The image in a form of a sensor_msgs::Image.
 */
void FaceDetectionTracker::imageCallback(const sensor_msgs::ImageConstPtr &msg)
{
    // Try to convert the message to cv image.
    try
    {
        m_cvPtr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);

        // Resize the image to half its size.
        //cv::resize(m_cvPtr->image, m_cvPtr->image, cv::Size(m_cvPtr->image.cols / 2, m_cvPtr->image.rows / 2));
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    // Apply the classifiers to the frame.
    if(m_cvPtr)
    {
        // ROS_INFO("Captured a frame!");
        detectAndDisplay(m_cvPtr->image);
        // We can always track when we have new image.
        track();
    }
    else
    {
        ROS_INFO("No captured frame!");
    }
}

/**
 * @brief      Function for detecting and displaying the faces.
 *
 * @param[in]  frame  The frame
 */
void FaceDetectionTracker::detectAndDisplay(cv::Mat frame)
{
    std::vector<Rect> faces;
    Mat frameGray;

    cv::cvtColor(frame, frameGray, CV_BGR2GRAY);
    //cv::equalizeHist(frameGray, frameGray);

    // Detect faces.
    //m_frontalfaceCascade.detectMultiScale(frameGray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
    m_frontalfaceCascade.detectMultiScale(frameGray, faces, 1.3, 3 );

    //for( size_t i = 0; i < faces.size(); i++ )
    if (faces.size() > 0)
    {
        // Center point
        // cv::Point center(faces[i].x + faces[i].width * 0.5, faces[i].y + faces[i].height * 0.5);

        // Point in the upper left corner.
        m_p1 = cv::Point(faces[i].x, faces[i].y);

        // Point in the lower right corner.
        m_p2 = cv::Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height);

        m_width = faces[i].width;
        m_height = faces[i].height;

        // Draw the rectangle on the frame.
        cv::rectangle(frameGray, m_p1, m_p2, Scalar(0, 0, 255), 4, 8, 0);

        // Signal a new bounding box.
        m_newBB_static = true;
    }

#ifdef DEBUG // Enable/Disable in the header.
    // Visualize the image with the fame.
    std::string box_text = format("# detected = %d", faces.size());
    cv::putText(frameGray, box_text, Point(10, 30), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
    cv::imshow( m_windowName, frameGray);
    cv::waitKey(3);
#endif
}

/////////////////////
/// Tracker part. ///
/////////////////////
/**
 * @brief      Track the face.
 */
void FaceDetectionTracker::track()
{

    // If new bounding box arrived (detected face) && we are not yet tracking anything.
    if (m_newBB_static && !tracking)
    {
        ROS_INFO("New bounding box!");
        // Create new tracker!
        cKCF = new cf_tracking::KcfTracker(m_paras);
        ROS_INFO("After bounding box!");
        // Save the incoming bounding box to a private member.
        bb.x = m_p1.x; // m_inBb.x;
        bb.y = m_p1.y; // m_inBb.y;
        bb.height = m_height; // m_inBb.height;
        bb.width = m_width; //m_inBb.width;

        // Reinitialize the tracker.
        if (cKCF->reinit(m_cvPtr->image, bb)) // KcfTracker->reinit(cv::Mat, cv::Rect)
        {
            // This means that it is correctly initalized.
            tracking = true;
            targetOnFrame = true;
        }
        else
        {
            // The tracker initialization has failed.
            delete cKCF;
            tracking = false;
            targetOnFrame = false;
        }
    }

    // If the target is on frame.
    if (targetOnFrame)
    {
        // Update the current tracker (if we have one)!
        targetOnFrame = cKCF->update(m_cvPtr->image, bb); 

        // If the tracking has been lost or the bounding box is out of limits.
        if (!targetOnFrame)
        {
            // We are not tracking anymore.
            delete cKCF;
            tracking = false;
        }
    }

    // If we are tracking, then publish the bounding box.
    if (tracking)
    {
        m_outBb.x = bb.x;
        m_outBb.y = bb.y;
        m_outBb.width = bb.width;
        m_outBb.height = bb.height;

        bbPub.publish(m_outBb);
    }


#ifdef DEBUG // Enable/Disable in the header.
    cv::Mat out_img;
    cv::cvtColor(m_cvPtr->image, out_img, CV_BGR2GRAY);// Convert to gray scale
    cv::cvtColor(out_img, out_img, CV_GRAY2BGR); //Convert from 1 color channel to 3 (trick)

    //Draw a rectangle on the out_img using the tracked bounding box.
    if (targetOnFrame)
    {
        cv::rectangle(out_img, cv::Point(bb.x, bb.y), cv::Point(bb.x + bb.width, bb.y + bb.height), cv::Scalar(0, 255, 255));
    }
    std::string box_text = format("# tracked = %d", -1);
    cv::putText(out_img, box_text, Point(10, 10), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);

    cv::imshow(m_windowName0, out_img);
    cv::waitKey(3);
#endif

    // Signal that the image and bounding box are not new.
    m_newBB_static = false;
}
