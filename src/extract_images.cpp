#include "extract_images.h"

FaceDetectionSave::FaceDetectionSave() : m_it(m_node)
{
    ///////////////////////
    /// Detection part. ///
    ///////////////////////

    //get camera topic
    m_node.getParam("/recording_sets/m_camera_topic", m_camera_topic);
    // Subscribe to input video feed and publish output video feed.
    m_imageSub = m_it.subscribe(m_camera_topic, 1, &FaceDetectionSave::imageCallback, this);


    //get location of haarcascade xml files
    m_node.getParam("/recording_sets/m_directory", m_directory);
    // Load the cascades.
    // Frontal face.
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
    m_node.getParam("/recording_sets/path_to_images", output_folder);
    skipFrames = 0;
    i=0;
}

FaceDetectionSave::~FaceDetectionSave()
{}

/////////////////////////////
/// Detecting and saving. ///
/////////////////////////////

void FaceDetectionSave::imageCallback(const sensor_msgs::ImageConstPtr &msg)
{
    // Convert the message to cv image.
    try
    {
        m_cvPtr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);

        // Resize the image to half its size if pictures are big
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
        // detection part
        detectAndDisplay(m_cvPtr->image);
    }
    else
    {
        ROS_INFO("No captured frame!");
    }
}


void FaceDetectionSave::detectAndDisplay(cv::Mat frame)
{
    std::vector<Rect> faces;
    Mat frameGray;

    cv::cvtColor(frame, frameGray, CV_BGR2GRAY);
    //Use histogram of equalization to attenuate illumination problem
    cv::equalizeHist(frameGray, frameGray);

    faceMethod = 0;

    if (skipFrames < 0 )
    {
        m_frontalfaceCascade.detectMultiScale(frameGray, faces, 1.3, 3 );
        if (faces.size() > 0)
        {
            faceMethod = 1;
        }
        /* before frontal does not work, we are not working with profiles
        if (faces.size() == 0)
        {
            m_profilefaceCascade.detectMultiScale(frameGray, faces, 1.3, 3 );
            if (faces.size() > 0)
            {
                faceMethod = 2;
            }
        }*/
        if (faces.size() > 0)
        {
            // face was just deteced on a picture, skipping next SKIP_FRAMES pictures
            // for CPU optimization   
            skipFrames = SKIP_FRAMES;
        }
        else
        {
            skipFrames--;
        }    
    }
    else
    {
        skipFrames--;
    }

    //for( size_t i = 0; i < faces.size(); i++ )
    if (faces.size() > 0)
    {
        // Point in the upper left corner.
        m_p1 = cv::Point(faces[0].x, faces[0].y);

        // Point in the lower right corner.
        m_p2 = cv::Point(faces[0].x + faces[0].width, faces[0].y + faces[0].height);

        m_width = faces[0].width;
        m_height = faces[i].height;

        cv::Rect R(m_p1, m_p2);
        face_for_save = frameGray(R);

        //ROS_INFO("frames size [%d, %d], dim %d", face_for_save.rows, face_for_save.cols, face_for_save.dims);
        cv::resize(face_for_save, face_for_save, cv::Size(FR_SI, FR_SI));
        //cv::cvtColor(frameGray, frameGray, CV_BGR2GRAY);
        imwrite(format("%s/eigenface_%d.png", output_folder.c_str(), i), norm_0_255(face_for_save));
        ROS_INFO("%d. face frame was saved", i);
        i++;
    }

#ifdef DEBUG // Enable/Disable in the header.
    cv::Mat out_img;
    // we should not edit the frame, because it is poiter
    frame.copyTo(out_img);
    // Visualize the image with the fame.
    
    switch(faceMethod)
    {
        case 2:
        { 
            std::string box_text = format("# profiles = %d", faces.size());
            cv::putText(out_img, box_text, Point(10, 10), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0); 
            cv::rectangle(out_img, m_p1, m_p2, CV_RGB(0, 255, 0), 4, 8, 0);
            break;
        }
        case 1:
        {
            std::string box_text = format("# frontal = %d", faces.size());
            cv::putText(out_img, box_text, Point(10, 10), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,255,0), 2.0);   
            cv::rectangle(out_img, m_p1, m_p2, CV_RGB(255, 255, 0), 4, 8, 0);
        } 
    }
    cv::imshow( m_windowName, out_img);
    cv::waitKey(3);
#endif
}

Mat FaceDetectionSave::norm_0_255(InputArray _src)
{
    Mat src = _src.getMat();
    // Create and return normalized image:
    Mat dst;
    switch(src.channels()) {
    case 1:
        cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
        break;
    case 3:
        cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
        break;
    default:
        src.copyTo(dst);
        break;
    }
    return dst;
}

