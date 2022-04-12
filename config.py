weightsPath = "yolo-coco/yolov3.weights"
configPath = "yolo-coco/yolov3.cfg"
inputVideoPath = "bridge.mp4"
outputVideoPath = "outputVideos/bridge.avi"
preDefinedConfidence = 0.5
preDefinedThreshold = 0.5
USE_GPU = False
run_every = 1
show_video = True
FRAMES_BEFORE_CURRENT = 10
inputWidth, inputHeight = 416, 416
horizontal_margin = 8
vertical_margin = 15
distance_margin = 20
BATCH_SIZE = 1