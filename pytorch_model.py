import time
from tracking import *
import os
import pandas as pd
from my_utils import *
import cv2
from setup_video import setup_video
import pickle
import torch

config_name = f"video_configs/{inputVideoPath.split('/')[-1]}_config"
# if not os.path.exists(config_name):
#     lanes, lines = setup_video(inputVideoPath, config_name)
# else:
#     lanes, lines = pickle.load(open(config_name, "rb"))
lanes, lines = [], []
np.random.seed(42)

all_detections = []
outputVideoPath = "batches_" + outputVideoPath
if os.path.exists(outputVideoPath):
    original_output_path = outputVideoPath.split('.')
    version = 1
    while os.path.exists(outputVideoPath):
        outputVideoPath = f"{original_output_path[0]}_{version}.avi"
        version += 1


# PURPOSE: Determining if the box-mid point cross the line or are within the range of 5 units
# from the line
# PARAMETERS: X Mid-Point of the box, Y mid-point of the box, Coordinates of the line
# RETURN:
# - True if the midpoint of the box overlaps with the line within a threshold of 5 units
# - False if the midpoint of the box lies outside the line and threshold
def boxAndLineOverlap(x_mid_point, y_mid_point, line_coordinates):
    x1_line, y1_line, x2_line, y2_line = line_coordinates  # Unpacking

    if (x_mid_point >= x1_line and x_mid_point <= x2_line + 5) and \
            (y_mid_point >= y1_line and y_mid_point <= y2_line + 5):
        return True
    return False


# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
# print("[INFO] loading YOLO from disk...")
# net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
#
# # Using GPU if flag is passed
# if USE_GPU:
#     net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
#     net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
#
# ln = net.getLayerNames()
# ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
model = torch.hub.load('ultralytics/yolov3', 'yolov3', pretrained=True)
# torch.save(model, "pytorch_model")
# model = torch.load("pytorch_model")
# initialize the video stream, pointer to output video file, and
# frame dimensions
videoStream = cv2.VideoCapture(inputVideoPath)
video_width = int(videoStream.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(videoStream.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Specifying coordinates for a default line
x1_line = 0
y1_line = video_height // 2
x2_line = video_width
y2_line = video_height // 2

# lines = [#Line((550,600),(1200,600), "Counting Line"),
# 		 Line((745, 1000), (800, 250), "Lane Divider 1"),
# 		 Line((1060, 1000), (940, 250), "Lane Divider 2")]

# Initialization
all_detections = [{(0, 0): 0} for i in range(FRAMES_BEFORE_CURRENT)]

# previous_frame_detections = [spatial.KDTree([(0,0)])]*FRAMES_BEFORE_CURRENT # Initializing all trees
num_frames, vehicle_count = 0, 0
frame_num = 0
writer = initializeVideoWriter(video_width, video_height, videoStream, outputVideoPath)
start_time = int(time.time())
# loop over frames from the video file stream
nframes = 2
# net.setInputsNames(["input"])
# net.setInputShape("input",(BATCH_SIZE,3,inputWidth, inputHeight))
while True:
    batch_start = time.time()
    frames = []
    for i in range(BATCH_SIZE):
        (grabbed, frame) = videoStream.read()
        frames.append(frame)
        frame_num += 1
    model_start = time.time()
    results = model(frames)
    results_dfs = results.pandas().xyxy
    results_df = results.pandas().xyxy[0]
    results_df.to_pickle("pytorch_output")
    if not grabbed:
        break
    print(f"FRAME:{frame_num}")


    # blob = cv2.dnn.blobFromImages(frames, 1 / 255.0, (inputWidth, inputHeight),
    # #                              swapRB=True, crop=False)
    # net.setInput(blob)
    start = time.time()
    # layerOutputs = net.forward(ln)
    # batch_outputs = [[el[j] for el in layerOutputs] for j in range(len(layerOutputs[0]))]
    model_time = time.time() - model_start
    print(model_time)
    # first_output = all_outputs[0]
    # end = time.time()
    for frame, output_df in zip(frames, results_dfs):
        id_start = time.time()
        # boxes, confidences, classids = parse_outputs(result,
        #                                              video_width,
        #                                              video_height)
        #

        output_df["width"] = (output_df.xmax - output_df.xmin).astype("int")
        output_df["height"] = (output_df.ymax - output_df.ymin).astype("int")
        x = list(output_df.xmin.astype("int"))
        y = list(output_df.ymin.astype("int"))
        width = list(output_df.width)
        height = list(output_df.height)
        boxes = list(zip(x, y, width, height))
        classids = list(output_df['class'])
        confidences = list(output_df.confidence)
        idxs, \
        vehicle_count, \
        current_detections, \
        current_positions = identify_vehicles(boxes,
                                              confidences,
                                              classids,
                                              frame,
                                              vehicle_count,
                                              all_detections)
        id_time = time.time() - id_start
        print("id ", id_time)
        #### LINES ####

    for line in lines:
        line.draw(frame)
        line.detect_crossings(current_positions)

    for lane in lanes:
        lane.draw(frame)
        lane.count_vehicles(current_positions)

        #### FRAME TEXT ####

    fps = BATCH_SIZE / ((time.time() - batch_start))
    frame_text = get_frame_text(lanes, lines, fps)
    write_text(frame, frame_text)

    #### SHOW AND SAVE FRAME ####
    writer.write(frame)
    all_detections.append(current_detections)
    df = pd.DataFrame([{label: center for center, label in prev.items()} for prev in all_detections])
    df.to_pickle('stats')
    if show_video:
        cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Updating with the current frame detections
    # previous_detections.pop(0)  # Removing the first frame from the list
    # # previous_frame_detections.append(spatial.KDTree(current_detections))
    # previous_detections.append(current_detections)


# release the file pointers
print("[INFO] cleaning up...")
df.to_pickle('stats')
writer.release()
videoStream.release()
