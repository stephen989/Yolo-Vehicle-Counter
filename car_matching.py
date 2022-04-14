import time
from tracking import *
import os
import pandas as pd
from my_utils import *
import cv2
from setup_video import setup_video
import pickle

config_name = f"video_configs/{inputVideoPath.split('/')[-1]}_config"
if not os.path.exists(config_name):
    lanes, lines = setup_video(inputVideoPath, config_name)
else:
    lanes, lines = pickle.load(open(config_name, "rb"))

lines = [Line(line.points, line.name) for line in lines]
lanes = [Lane(lane.points, lane.name) for lane in lanes]
np.random.seed(42)

all_detections = []

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
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# Using GPU if flag is passed
if USE_GPU:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

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
while True:
    frame_start = time.time()

    for i in range(run_every):
        (grabbed, frame) = videoStream.read()
        frame_num += 1
    if not grabbed:
        break
    print(f"FRAME:{frame_num}")

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (inputWidth, inputHeight),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    # start = time.time()
    layerOutputs = net.forward(ln)
    # end = time.time()
    boxes, confidences, classids = parse_outputs(layerOutputs,
                                                 video_width,
                                                 video_height)

    idxs, \
    vehicle_count, \
    current_detections, \
    current_positions = identify_vehicles(boxes,
                                          confidences,
                                          classids,
                                          frame,
                                          vehicle_count,
                                          all_detections)
    #### LINES ####

    for line in lines:
        line.draw(frame)
        line.detect_crossings(current_positions)
        # line.label(frame)

    for lane in lanes:
        lane.draw(frame)
        # lane.label(frame)
        lane.count_vehicles(current_positions)

    #### FRAME TEXT ####

    fps = 1 / (time.time() - frame_start)
    frame_text = get_frame_text(lanes, lines, fps)
    write_text(frame, frame_text)

    #### SHOW AND SAVE FRAME ####
    writer.write(frame)
    if show_video:
        cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Updating with the current frame detections
    # previous_detections.pop(0)  # Removing the first frame from the list
    # # previous_frame_detections.append(spatial.KDTree(current_detections))
    # previous_detections.append(current_detections)
    all_detections.append(current_detections)
    df = pd.DataFrame([{label: center for center, label in prev.items()} for prev in all_detections])
    df.to_pickle('stats')

# release the file pointers
lane_log = create_lane_log(lanes)
lane_switches = get_switches(lane_log)
pickle_dump(lane_switches, "switches")
print(lane_switches)
print("[INFO] cleaning up...")
df.to_pickle('stats')
pickle_dump(lanes, "lanes")
pickle_dump(lines, "lines")
writer.release()
videoStream.release()
