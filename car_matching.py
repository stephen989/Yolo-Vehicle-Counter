import numpy as np
# import imutils
import time
import cv2
from scipy import spatial
# from video_player import *
import os
import pandas as pd
from utils import *
import cv2
np.random.seed(42)
# from input_retrieval import *
# All these classes will be counted as 'vehicles'
# Setting the threshold for the number of frames to search a vehicle for


# Parse command line arguments and extract the values required



all_positions = []

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


def dist(x1, x2):
    return np.sum((x1-x2)**2, axis = 1)


def single_dist(x1, x2):
    return np.sum((x1-x2)**2)


def tiebreak(prev_labels, centers, labels, vehicle_count, current_detections):
    for label in set(labels):
        ixs = list(np.where(np.array(labels)==label)[0])
        if len(ixs) > 1:
            distances = [single_dist(np.array(prev_labels[label]), np.array(centers[ix])) for ix in ixs]
            closest = np.argmin(distances)
            labels[ixs[closest]] = label
            ixs.pop(closest)
            for ix in ixs:
                labels[ix] = vehicle_count
                current_detections[tuple(centers[ix])] = vehicle_count
                vehicle_count += 1
    return labels, vehicle_count, current_detections

def match_labels(previous_detections, current_detections, vehicle_count):
    most_recent_positions = get_most_recent_position(previous_detections)
    prev_centers = np.array(list(most_recent_positions.keys()))
    centers = np.array(list(current_detections.keys()))
    labels = []
    for center in centers:
        closest_i = np.argmax(-1*dist(center, prev_centers))
        closest = prev_centers[closest_i]
        labels.append(most_recent_positions[(closest[0], closest[1])])
        current_detections[(center[0], center[1])] = most_recent_positions[tuple(prev_centers[closest_i])]
    prev_labels = {label:center for center, label in most_recent_positions.items()}
    labels, vehicle_count, current_detections = tiebreak(prev_labels, centers, labels, vehicle_count,  current_detections)
    return labels, vehicle_count, current_detections


def count_vehicles(idxs, boxes, classIDs, vehicle_count, previous_frame_detections, frame):
    current_detections = {}
    curr_vehicle_count = 0
    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indices we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            centerX = x + (w // 2)
            centerY = y + (h // 2)

            # When the detection is in the list of vehicles, AND
            # it crosses the line AND
            # the ID of the detection is not present in the vehicles
            if (LABELS[classIDs[i]] in list_of_vehicles):
                current_detections[(centerX, centerY)] = curr_vehicle_count
                # curr_vehicle_count +=1
        labels, vehicle_count, current_detections = match_labels(previous_frame_detections, current_detections, vehicle_count)

        for (x, y) in current_detections.keys():
            id = current_detections.get((x, y))
            cv2.putText(frame, str(id), (x, y), \
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 255], 2)

    return vehicle_count, current_detections

def get_most_recent_position(previous_frame_detections):
    most_recent_positions = {}
    for detections in previous_frame_detections:
        prev_labels = {label:center for center, label in detections.items()}
        for label in prev_labels.keys():
            most_recent_positions[label] = prev_labels[label]
    most_recent_positions = {label:center for center, label in most_recent_positions.items()}
    return most_recent_positions


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
lines = [Line((550,600),(1200,600), "Counting Line"),
		 Line((745, 1000), (800, 250), "Lane Divider 1"),
		 Line((1060, 1000), (940, 250), "Lane Divider 2")]

# Initialization
previous_frame_detections = [{(0, 0): 0} for i in range(FRAMES_BEFORE_CURRENT)]

# previous_frame_detections = [spatial.KDTree([(0,0)])]*FRAMES_BEFORE_CURRENT # Initializing all trees
num_frames, vehicle_count = 0, 0
frame_num = 0
writer = initializeVideoWriter(video_width, video_height, videoStream, outputVideoPath)
start_time = int(time.time())
# loop over frames from the video file stream
while True:
    frame_start = time.time()
    frame_text = ""
    frame_num += 1
    (grabbed, frame) = videoStream.read()
    if not grabbed:
        break

    if frame_num % show_every != 0:
        if frame_num > show_every:
            drawDetectionBoxes(idxs, boxes, classIDs, confidences, frame)
            displayVehicleCount(frame, vehicle_count)
            for line in lines:
                line.draw(frame)
            # writer.write(frame)

        continue


    print(f"FRAME:{frame_num}")
    # Initialization for each iteration
    boxes, confidences, classIDs = [], [], []
    # Calculating fps each second
    # start_time, num_frames = displayFPS(start_time, frame_num)
    # construct a blob from the input frame and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes
    # and associated probabilities
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (inputWidth, inputHeight),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    # start = time.time()
    layerOutputs = net.forward(ln)
    # end = time.time()

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for i, detection in enumerate(output):
            # extract the class ID and confidence (i.e., probability)
            # of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > preDefinedConfidence:
                # scale the bounding box coordinates back relative to
                # the size of the image, keeping in mind that YOLO
                # actually returns the center (x, y)-coordinates of
                # the bounding box followed by the boxes' width and
                # height
                box = detection[0:4] * np.array([video_width, video_height, video_width, video_height])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top
                # and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # Printing the info of the detection
                # print('\nName:\t', LABELS[classID],
                # '\t|\tBOX:\t', x,y)

                # update our list of bounding box coordinates,
                # confidences, and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping
    # bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, preDefinedThreshold,
                            0.5)

    # Draw detection box
    drawDetectionBoxes(idxs, boxes, classIDs, confidences, frame)

    vehicle_count, current_detections = count_vehicles(idxs, boxes, classIDs, vehicle_count, previous_frame_detections,
                                                       frame)
    # print(vehicle_count, current_detections)
    current_positions = {label:center for center, label in current_detections.items()}
    # Display Vehicle Count if a vehicle has passed the line
    # displayVehicleCount(frame, vehicle_count)
    frame_text += f"Vehicle count: {len(lines[0].crossings)}"
    for line in lines:
        line.draw(frame)
        line.detect_crossings(current_positions)
        frame_text += f"\n{line.name}: {len(line.crossings)}"
    lane_switches = sum([len(line.crossings) for line in lines if "Lane" in line.name])
    frame_text += f"\nLane Switches: {lane_switches}"
    frame_time = time.time() - frame_start
    frame_text += f"\nTime per frame: {frame_time:.2f}s"
    write_text(frame, frame_text)


    # write the output frame to disk
    for i in range(show_every):
        writer.write(frame)
    if show_video:
        cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Updating with the current frame detections
    previous_frame_detections.pop(0)  # Removing the first frame from the list
    # previous_frame_detections.append(spatial.KDTree(current_detections))
    previous_frame_detections.append(current_detections)
    all_positions.append(current_detections)
    df = pd.DataFrame([{label: center for center, label in prev.items()} for prev in all_positions])
    df.to_pickle('stats')

# release the file pointers
print("[INFO] cleaning up...")
df = pd.DataFrame([{label:center for center, label in prev.items()} for prev in all_positions])
df.to_pickle('stats')
writer.release()
videoStream.release()
