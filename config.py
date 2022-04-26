MODEL_TYPE = "pytorch"
weightsPath= "yolo-coco/yolov3.weights"
configPath = "yolo-coco/yolov3.cfg"
inputVideoPath = "test_videosbridge.mp4"
outputVideoPath = "outputVideos/nyc.mp4"
preDefinedConfidence = 0.5
preDefinedThreshold = 0.5
USE_GPU = False
RUN_EVERY = 1
show_video = True
FRAMES_BEFORE_CURRENT = 5
inputWidth, inputHeight = 416, 416
horizontal_margin = 8
vertical_margin = 15
distance_margin = 20
BATCH_SIZE = 32