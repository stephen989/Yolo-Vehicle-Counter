import time
from tracking import *
import os
import pandas as pd
from my_utils import *
import cv2
from setup_video import setup_video
import pickle
from choose_models import *

plt.style.use('fivethirtyeight')

np.random.seed(42)
fig = plt.figure()

if os.path.exists(outputVideoPath):
    original_output_path = outputVideoPath.split('.')
    version = 1
    while os.path.exists(outputVideoPath):
        outputVideoPath = f"{original_output_path[0]}_dashcam_{version}.avi"
        version += 1

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

# Initialization
all_detections = [{(0, 0): 0} for i in range(FRAMES_BEFORE_CURRENT)]

# previous_frame_detections = [spatial.KDTree([(0,0)])]*FRAMES_BEFORE_CURRENT # Initializing all trees
num_frames, vehicle_count = 0, 0
frame_num = 0
writer = initializeVideoWriter(video_width, video_height, videoStream, outputVideoPath)
start_time = int(time.time())
# loop over frames from the video file stream
while True:
    batch_start = time.time()
    frames = []

    for i in range(BATCH_SIZE):
        for j in range(RUN_EVERY):
            (grabbed, frame) = videoStream.read()
            frame_num += 1
        if frame is None:
            break
        frames.append(frame)
    if not frames:
        break

    print(f"FRAME:{frame_num}")
    boxes_list, confidences_list, classes_list = get_model_output(model, frames)




    for boxes, confidences, classids, frame in zip(boxes_list, confidences_list, classes_list, frames):
        draw_dashcam_boxes(boxes, classids, confidences, frame)
        # idxs, \
        # vehicle_count, \
        # current_detections, \
        # current_positions = identify_vehicles(boxes,
        #                                       confidences,
        #                                       classids,
        #                                       frame,
        #                                       vehicle_count,
        #                                       all_detections)
        # all_detections.append(current_detections)


        #### FRAME TEXT ####
        # fps = BATCH_SIZE / (time.time() - batch_start)
        # frame_text = get_frame_text(lanes, lines, fps)
        # write_text(frame, frame_text)

        #### SHOW AND SAVE FRAME ####
        writer.write(frame)


    # df = pd.DataFrame([{label: center for center, label in prev.items()} for prev in all_detections])
    # df.to_pickle(f'stats_dfs/{outputVideoPath.split("/")[-1].split(".")[-2]}')
    # counts = get_counts(lanes, df)
    if show_video and frame_num > 2 * RUN_EVERY:
        # plt.xlim((0, frame_num / RUN_EVERY))
        # fig, img = create_plot_img(fig, counts)
        # cv2.imshow("Plot", img)
        cv2.imshow("Video Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if frame_num>120:
        break

# release the file pointers
# max_id = max([max(list(detections.values())) for detections in all_detections])
# lane_log = create_lane_log(lanes, max_id)
# lane_switches = get_switches(lane_log)
# pickle_dump(lane_switches, "switches")
# print(lane_switches)
# print("[INFO] cleaning up...")
# df.to_pickle('stats')
# pickle_dump(lanes, "lanes")
# pickle_dump(lines, "lines")
# writer.release()
# videoStream.release()
