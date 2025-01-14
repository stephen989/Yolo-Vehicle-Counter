import time
from tracking import *
import os
import pandas as pd
from my_utils import *
import cv2
from setup_video import setup_video
import pickle
plt.style.use('fivethirtyeight')

config_name = f"video_configs/{inputVideoPath.split('/')[-1]}_config"
if not os.path.exists(config_name):
    lanes, lines = setup_video(inputVideoPath, config_name)
else:
    lanes, lines = pickle.load(open(config_name, "rb"))

lines = [Line(line.points, line.name) for line in lines]
lanes = [Lane(lane.points, lane.name) for lane in lanes]
np.random.seed(42)
fig = plt.figure()
all_detections = []

if os.path.exists(outputVideoPath):
    original_output_path = outputVideoPath.split('.')
    version = 1
    while os.path.exists(outputVideoPath):
        outputVideoPath = f"{original_output_path[0]}_{version}.avi"
        version += 1



print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)



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

    for i in range(RUN_EVERY):
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
    for line in lines:
        line.draw(frame)
    for lane in lanes:
        lane.draw(frame)

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
        # line.draw(frame)
        line.detect_crossings(current_positions)
        # line.label(frame)

    for lane in lanes:
        # lane.draw(frame)
        # lane.label(frame)
        lane.count_vehicles(current_positions)

    #### FRAME TEXT ####

    fps = 1 / (time.time() - frame_start)
    frame_text = get_frame_text(lanes, lines, fps)
    write_text(frame, frame_text)

    #### SHOW AND SAVE FRAME ####
    writer.write(frame)

    all_detections.append(current_detections)
    df = pd.DataFrame([{label: center for center, label in prev.items()} for prev in all_detections])
    df.to_pickle(f'stats_dfs/{outputVideoPath.split("/")[-1].split(".")[-2]}')
    counts = get_counts(lanes, df)
    if show_video and frame_num > 2 * RUN_EVERY:
        plt.xlim((0, frame_num / RUN_EVERY))
        fig, img = create_plot_img(fig, counts)
        cv2.imshow("Plot", img)
        cv2.imshow("Video Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break




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
