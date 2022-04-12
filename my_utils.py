import cv2
import numpy as np
from config import *
import matplotlib.path as mpltPath



LABELS = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
          'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
          'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
          'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
          'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
          'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa',
          'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard',
          'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
          'teddy bear', 'hair drier', 'toothbrush']
np.random.seed(1)
list_of_vehicles = ["bicycle", "car", "motorbike", "bus", "truck", "train"]
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                           dtype="uint8")


def get_eqn(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    A = (y2-y1)/(x2-x1)
    B = -1
    c = y1-A*x1
    return A, B, c


def distance_to_line(A, B, c, px, py):
    return abs(A*px + B*py + c)/np.sqrt(A**2 + B**2)


class Lane:
    def __init__(self, points, name="Lane"):
        self.points = points.reshape(len(points), 2)
        self.vehicles = 0
        self.name = name
        self.path = mpltPath.Path(self.points)
        self.count = 0

    def draw(self, frame, color=(0, 0, 0)):
        color = np.random.randint(0, 255, 3)
        frame = cv2.polylines(frame,
                              [self.points.astype(np.int32)],
                              isClosed=True,
                              color=(0, 0, 255),
                              thickness=3)
        return frame

    def count_vehicles(self, current_positions):
        self.count = sum(self.path.contains_points(list(current_positions.values())))
        return self.count



    def label(self, frame):
        y = 0.5 * (self.points.max(0)[0][1] + self.points.min(0)[0][1])
        x = 0.5 * (self.points.min(0)[0][0] + self.points.max(0)[0][0])
        write_text(frame, self.name, (x, y), anchor="center")




class Line:
    def __init__(self, points, name="Line"):
        self.point1, self.point2 = points.reshape(2, 2)
        self.points = points.reshape(2, 2)
        x1_temp, y1_temp = self.point1
        x2_temp, y2_temp = self.point2
        self.x1 = min(x1_temp, x2_temp)
        self.x2 = max(x1_temp, x2_temp)
        self.y1 = min(y1_temp, y2_temp)
        self.y2 = max(y1_temp, y2_temp)
        self.A, self.B, self.c = get_eqn(self.point1, self.point2)
        self.crossings = []
        self.name = name

    def draw(self, image, color = (0,0,0)):
        cv2.line(image, tuple(self.point1.astype(np.int32)), tuple(self.point2.astype(np.int32)), color, 2)

    def detect_crossings(self, current_positions):
        new_crossings = []
        for vehicle in current_positions.keys():
            position = current_positions[vehicle]
            if (position[0] + horizontal_margin > self.x1) and ((position[0] - horizontal_margin < self.x2)):
                if (position[1] + vertical_margin > self.y1) and ((position[1] - vertical_margin < self.y2)):
                    if distance_to_line(self.A, self.B, self.c, position[0], position[1]) < distance_margin:
                        new_crossings.append(vehicle)
        new_crossings = [crossing for crossing in new_crossings if crossing not in self.crossings]
        if new_crossings:
            print(f"{self.name} crossings: ", " ".join([str(vehicle) for vehicle in new_crossings]))
        self.crossings += new_crossings

    def label(self, frame):
        y = np.mean(self.points[:,1]) + 40
        x = np.min(self.points[:,0])
        write_text(frame, self.name, (x, y))


# PURPOSE: Displays the vehicle count on the top-left corner of the frame
# PARAMETERS: Frame on which the count is displayed, the count number of vehicles
# RETURN: N/A
def displayVehicleCount(frame, vehicle_count):
    cv2.putText(
        frame,  # Image
        'Detected Vehicles: ' + str(vehicle_count),  # Label
        (20, 20),  # Position
        cv2.FONT_HERSHEY_SIMPLEX,  # Font
        0.8,  # Size
        (0, 0xFF, 0),  # Color
        2,  # Thickness
        cv2.FONT_HERSHEY_COMPLEX_SMALL,
    )

# PURPOSE: Displaying the FPS of the detected video
# PARAMETERS: Start time of the frame, number of frames within the same second
# RETURN: New start time, new number of frames
def displayFPS(start_time, num_frames):
    current_time = int(time.time())
    if (current_time > start_time):
        # os.system('clear') # Equivalent of CTRL+L on the terminal
        print("FPS:", num_frames)
        num_frames = 0
        start_time = current_time
    return start_time, num_frame

# PURPOSE: Draw all the detection boxes with a green dot at the center
# RETURN: N/A
def drawDetectionBoxes(idxs, boxes, classIDs, confidences, frame):
    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indices we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the frame
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]],
                                       confidences[i])
            cv2.putText(frame, text, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            # Draw a green dot in the middle of the box
            cv2.circle(frame, (x + (w // 2), y + (h // 2)), 2, (0, 0xFF, 0), thickness=2)


def write_text(frame,
               lines,
               position = (20, 20),
               anchor = "left",
               padding = 5,
               bg=(255,255,255)):
    lines = lines.split("\n")
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.75
    font_color = (0, 0, 0)
    font_thickness = 8

    text_sizes = [cv2.getTextSize(text, font, font_scale, font_thickness)[0] for text in lines]
    total_height = sum([height for (width, height) in text_sizes]) + padding * (len(lines) + 1)
    total_width = max([width for (width, height) in text_sizes]) + padding + 20
    if anchor == "left":
        x, y = position
    else:
        x, y = position
        x -= 0.75*total_width
    x, y = int(x), int(y)
    cv2.rectangle(frame, (x, y + total_height), (x + total_width, y-padding), bg, -1)


    for text in lines:
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_w, text_h = text_size
        # cv2.rectangle(frame, (x, y-text_h), (x + text_w, y - text_h), bg, -1)
        cv2.putText(frame,
                    text.upper(),
                    (x, y+text_h),
                    font,
                    font_scale,
                    font_color)
        y += text_h + padding

# PURPOSE: Initializing the video writer with the output video path and the same number
# of fps, width and height as the source video
# PARAMETERS: Width of the source video, Height of the source video, the video stream
# RETURN: The initialized video writer
def initializeVideoWriter(video_width, video_height, videoStream, outputVideoPath):
    # Getting the fps of the source video
    sourceVideofps = videoStream.get(cv2.CAP_PROP_FPS)
    # initialize our video writer
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    return cv2.VideoWriter(outputVideoPath, fourcc, sourceVideofps,
                           (video_width, video_height), True)


def get_frame_text(lanes, lines, fps):
    frame_text = ""
    for i, line in enumerate(lines):
      frame_text += f"\n{line.name} {str(i)}: {len(line.crossings)}"
    for i, lane in enumerate(lanes):
      frame_text += f"\n{lane.name} {str(i)}: {lane.count}"
    nlane_switches = sum([len(line.crossings) for line in lines if "Lane" in line.name])
    frame_text += f"\nLane Switches: {nlane_switches}"
    frame_text += f"\nFPS: {fps:.2f}"
    return frame_text
#