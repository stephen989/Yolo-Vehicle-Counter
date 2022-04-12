import pickle
from my_utils import *

inputVideoPath = "test_videos/dundrum.mp4"
lanes, lines = pickle.load(open("test_config", "rb"))
lanes = [Lane(lane.points, lane.name) for lane in lanes]
lines = [Line(line.points, line.name) for line in lines]
videoStream = cv2.VideoCapture(inputVideoPath)
(grabbed, frame) = videoStream.read()

pts = np.zeros((4, 2), np.int32)

counter = 0



while True:
    # Showing original image
    for lane in lanes:
        frame = lane.draw(frame)
        lane.label(frame)
    for line in lines:
        line.label(frame)
        line.draw(frame)
    cv2.imshow("Original Image ", frame)
    if cv2.waitKey(20) & 0xFF == 27:
        break

    # Mouse click event on original image
    # Printing updated point matrix
    # print(point_matrix)
    # Refreshing window all time
    cv2.waitKey(1)