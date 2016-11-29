#include "face_detection_tracker.h"

int main(int argc, char** argv)
{
    ros::init(argc, argv, "face_detection_tracker");

    /** @face_detection_tracker */
    FaceDetectionTracker fd;

    ROS_INFO("initialized the class");

    while (ros::ok())
    {
        ros::spinOnce();
        // we are running tracker when we have new image
        //fd.track();
    }

    return 0;
}
