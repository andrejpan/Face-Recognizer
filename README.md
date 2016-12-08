# Face Recognizer implementation with OpenCV and ROS

Recognizing will work if just one person is on the image. System will have unexpected results when there are multiple faces on images.

Package was tested with Ubuntu 14.04, ROS Indigo and OpenCV 2.4

Video: https://www.youtube.com/watch?v=3dSMyXtXyfU

## Brief description.
- Package subscribes to video topic.
- Detector firstly finds a face on a image. Detecting is not happening on every frame. Video stream, 30Hz, resolution 640x480px takes 85-90% off all 8 cores on `Intel® Core™ i7 CPU 930 @ 2.80GHz`. Face detection is based [Cascade Classification](http://docs.opencv.org/2.4/modules/objdetect/doc/cascade_classification.html). Other files for detecting different body parts are located [here](https://github.com/opencv/opencv/tree/master/data).
- When detector finds a face it sends data to tracker. Tracker is based on this [project](https://github.com/klahaag/cf_tracking).
- Face Recognition with OpenCV is based on OpenCV [project](http://docs.opencv.org/2.4/modules/contrib/doc/facerec/facerec_tutorial.html). Recognition is happening when face is detected.

## Before installation
- install ROS: http://wiki.ros.org/ROS/Installation, full version is advised (OpenCV is included). 
- `git clone https://github.com/andrejpan/Face-Recognizer.git`
- `cd Face-Recognizer`
- Package is using custom message which is included in the package [`perception_msgs`](perception_msgs). This package must be build before compiling Face recognizer. `mv perception_msgs ..`
- `catkin build perception_msgs`
- Source it.

## Package installation 
- `catkin build face_detection_tracker`
- Source it.

## Acquiring images for training operation
- Before you can run Face recognizer you need to record sets of images for training operation. 
- Rule of thumb: record around 100 images per one person in different positions and remove bad ones (face covered by hand or something similar). 
- Advising to use same camera for taking training images and also for running a Face recognizer.
- Size of face images is `100x100px`. 
- Modify configuration file: `configs/param.yaml`
- Running: `roslaunch face_detection_tracker extract.launch`
- Arrange images in separate folders, each person in separate folder (person0, person1,..,). At `src/face_detection_tracker.cpp` you can name people inside `my_map`.

## Run Face Recognizer
- After having all sets you can run `Face Recognizer`. All images for training operation have to be the same size.
- `roslaunch face_detection_tracker face_recognizer.launch`

## TODO
- remove perception_msgs with standard ros 3D point
- kill and revive tracker on every few seconds
- read all folders of faces in given path
- add support for multiple faces on images.
- detecting unknown faces - removing false positives
- update the system: http://stackoverflow.com/questions/21339307/how-to-get-better-results-with-opencv-face-recognition-module
