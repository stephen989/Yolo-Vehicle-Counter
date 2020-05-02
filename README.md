# Yolo Vehicle Counter

## Overview
You Only Look Once (YOLO) is a CNN architecture for performing real-time object detection. The algorithm applies a single neural network to the full image, and then divides the image into regions and predicts bounding boxes and probabilities for each region. For more detailed working of YOLO algorithm, please refer to the [YOLO paper](https://pjreddie.com/media/files/papers/YOLOv3.pdf). 

This project aims to count every vehicle(motorcycle, bus, car, cycle, truck) detected in the input video using YOLOv3 object-detection algorithm.

## Working 
As shown in the [gif], the vehicles crossing the red line are counted. When the center point of the detection box of the vehicle (green dot) intersects with the line, the vehicle counter increments by one.  

## Prerequisites
* Linux distro (Tested on Ubuntu 18.04)
* A street video file to run the vehicle counting 
* The pre-trained yolov3 weight file should be downloaded by following these steps:
```
cd yolo-coco
wget https://pjreddie.com/media/files/yolov3.weights
``` 

## Dependencies
* OpenCV 3.4 or above(Tested on OpenCV 3.4.2.17)
```
pip3 install opencv-python==3.4.2.17
```
* Python3 (Tested on Python 3.6.9)
```
sudo apt-get upgrade python3
```
* Imutils 
```
pip3 install imutils
```

## Usage
```
python yolo_video.py --input <input video path> --output <output video path> --yolo yolo-coco
```
Example: 
```
python yolo_video.py --input inputVideos/highway.mp4 --output outputVideos/highwayOut.avi --yolo yolo-coco
```
## Reference
* https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/ 
