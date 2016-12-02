#include "extract_images.h"

int main(int argc, char** argv)
{
    ros::init(argc, argv, "face_detection_and_save_pictures");

    FaceDetectionSave fdt;

    ROS_INFO("initialized the class");

    while (ros::ok())
    {
        ros::spinOnce();
    }

    return 0;
}
