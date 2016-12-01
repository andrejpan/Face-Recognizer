#include "face_detection_tracker.h"

// Initialize static members.
bool FaceDetectionTracker::m_newBB_static = false;

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
    skipFrames = 0;

    /////////////////////
    /// Tracker part. ///
    /////////////////////

    bbPub = m_node.advertise<perception_msgs::Rect>("tracker/bb", m_queuesize);

    m_paras.enableTrackingLossDetection = true;
    // paras.psrThreshold = 10; // lower more flexible
    m_paras.psrThreshold = 13.5; // higher more restricted to changes
    m_paras.psrPeakDel = 2; // 1;

    timeout = ros::Duration(2);

    // for adding faces to vector
    num_images = 0;
    //model = createFisherFaceRecognizer();
    model = createEigenFaceRecognizer();
    min_frame_size.x = 4000; min_frame_size.y = 4000;  
}

FaceDetectionTracker::~FaceDetectionTracker()
{}

///////////////////////
/// Detection part. ///
///////////////////////

void FaceDetectionTracker::imageCallback(const sensor_msgs::ImageConstPtr &msg)
{
    // Convert the message to cv image.
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
        // detection part
        detectAndDisplay(m_cvPtr->image);
        // We can always track when we have new image.
        track();
    }
    else
    {
        ROS_INFO("No captured frame!");
    }
}


void FaceDetectionTracker::detectAndDisplay(cv::Mat frame)
{
    std::vector<Rect> faces;
    Mat frameGray;

    cv::cvtColor(frame, frameGray, CV_BGR2GRAY);
    //cv::equalizeHist(frameGray, frameGray);

    faceMethod = 0;

    if (skipFrames < 0 )
    {
        m_frontalfaceCascade.detectMultiScale(frameGray, faces, 1.3, 3 );
        if (faces.size() > 0)
        {
            faceMethod = 1;
        }
        /*
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

    //ROS_INFO("%d ", faceMethod);
    //for( size_t i = 0; i < faces.size(); i++ )
    i=0;
    if (faces.size() > 0)
    {
        // Point in the upper left corner.
        m_p1 = cv::Point(faces[i].x, faces[i].y);

        // Point in the lower right corner.
        m_p2 = cv::Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height);

        m_width = faces[i].width;
        m_height = faces[i].height;

        // Signal a new bounding box.
        m_newBB_static = true;

        // add faces to the list
        if (num_images < VEC_SIZE)
        {
            cv::Rect R(m_p1, m_p2); //Create a rect 
            images.push_back(frame(R));
            labels.push_back(0);
            num_images++;
            //ROS_INFO("Vector size %d", num_images);
            //ROS_INFO("Frame size [%d, %d]", m_height, m_width);
            if (min_frame_size.x > m_width)
                min_frame_size.x = m_width;
            if (min_frame_size.y > m_height)
                min_frame_size.y = m_height;
        }
        if (num_images == VEC_SIZE)
        {
            // pictures need to be resized to the same size
            ROS_INFO("Min frame size [%d, %d]", min_frame_size.y, min_frame_size.x);
            int tmpi = 0;
            for (std::vector<cv::Mat>::iterator it = images.begin() ; it != images.end(); ++it)
            {
                //cv::Mat tmpMat = *it;
                cv::resize(*it, *it, cv::Size(min_frame_size.x, min_frame_size.y));
                cv::cvtColor(*it, *it, CV_BGR2GRAY);
                //ROS_INFO("%d", it->rows);
                ROS_INFO("frames size [%d, %d], dim %d", it->rows, it->cols, it->dims);
                //imshow(format("pic %d", tmpi), *it);
                tmpi++;
            }

            ros::Time tic = ros::Time::now();
            ROS_INFO("before training step");
            model->train(images, labels);
            ros::Time toc = ros::Time::now();
            ROS_INFO("Time lasted for training %f", (toc-tic).toSec() );

            // avoiding running training step again
            num_images++;

            // Here is how to get the eigenvalues of this Eigenfaces model:
            cv::Mat eigenvalues = model->getMat("eigenvalues");
            // And we can do the same to display the Eigenvectors (read Eigenfaces):
            cv::Mat W = model->getMat("eigenvectors");
            // Get the sample mean from the training data
            cv::Mat mean = model->getMat("mean");

            imshow("mean", norm_0_255(mean.reshape(1, images[0].rows)));
            // Display or save the Eigenfaces:
            for (int i = 0; i < min(10, W.cols); i++)
            {
                string msg = format("Eigenvalue #%d = %.5f", i, eigenvalues.at<double>(i));
                std::cout << msg << std::endl;
                // get eigenvector #i
                Mat ev = W.col(i).clone();
                // Reshape to original size & normalize to [0...255] for imshow.
                Mat grayscale = norm_0_255(ev.reshape(1,  min_frame_size.x));
                // Show the image & apply a Jet colormap for better sensing.
                Mat cgrayscale;
                cv::applyColorMap(grayscale, cgrayscale, COLORMAP_JET);
                // Display or save:
                imshow(format("eigenface_%d", i), cgrayscale);
            }
            // Display or save the image reconstruction at some predefined steps:
            ROS_INFO("W size [%d, %d]", W.cols, W.rows);
            for(int num_components = min(W.cols, 10); num_components < min(W.cols, 300); num_components+=15)
            {
                // slice the eigenvectors from the model
                Mat evs = Mat(W, Range::all(), Range(0, num_components));
                Mat projection = subspaceProject(evs, mean, images[0].reshape(1,1));
                Mat reconstruction = subspaceReconstruct(evs, mean, projection);
                // Normalize the result:
                reconstruction = norm_0_255(reconstruction.reshape(1, images[0].rows));
                // Display or save:
                imshow(format("eigenface_reconstruction_%d", num_components), reconstruction);
            }

        }

        if(num_images > VEC_SIZE)
        {
            cv::Rect R(m_p1, m_p2); 
            cv::Mat tmpMat;
            cv::resize(frame(R), tmpMat, cv::Size(min_frame_size.x, min_frame_size.y));
            cv::cvtColor(tmpMat, tmpMat, CV_BGR2GRAY);
            //ROS_INFO("just loaded frames size [%d, %d], dim %d", tmpMat.rows, tmpMat.cols, tmpMat.dims);
            int predictedLabel = model->predict(tmpMat);
            ROS_INFO("Predicted class = %d / Actual class = %d.", predictedLabel, 0);
        }
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
    cv::waitKey(33);
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
        // Save the incoming bounding box to a private member.
        bb.x = m_p1.x; 
        bb.y = m_p1.y; 
        bb.height = m_height; 
        bb.width = m_width; 

        // Reinitialize the tracker.
        if (cKCF->reinit(m_cvPtr->image, bb)) // KcfTracker->reinit(cv::Mat, cv::Rect)
        {
            // This means that it is correctly initalized.
            tracking = true;
            targetOnFrame = true;
            start_time = ros::Time::now();
        }
        else
        {
            // The tracker initialization has failed.
            delete cKCF;
            tracking = false;
            targetOnFrame = false;
            skipFrames = -1;
        }
    }

    // If the target is on frame.
    if (targetOnFrame)
    {
        /*bb.x = m_p1.x;
        bb.y = m_p2.y; 
        bb.width = m_width; 
        bb.height = m_height;*/ 
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
    
    if (ros::Time::now() - start_time > timeout) 
    {
            //ROS_INFO("Just reseted****************************************************");
            start_time = ros::Time::now();
            // we need to implement reseting the window without a long delay.
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
        cv::rectangle(out_img, cv::Point(bb.x, bb.y), cv::Point(bb.x + bb.width, bb.y + bb.height), CV_RGB(255, 0, 0));
    }
    std::string box_text = format("# tracked = %d", -1);
    cv::putText(out_img, box_text, Point(10, 10), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,0,0), 2.0);

    cv::imshow(m_windowName0, out_img);
    cv::waitKey(3);
#endif

    // Signal that the image and bounding box are not new.
    m_newBB_static = false;
}


Mat FaceDetectionTracker::norm_0_255(InputArray _src) {
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
