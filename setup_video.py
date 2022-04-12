import cv2
import numpy as np
import pickle
from my_utils import Line, Lane
import tkinter as tk


class TextInput:
    def __init__(self):
        self.frame = tk.Tk()
        self.frame.title("TextBox Input")
        self.frame.geometry('400x200')
        self.inputtxt = tk.Text(self.frame,
                                   height=5,
                                   width=20)
        self.inputtxt.pack()
        saveButton = tk.Button(self.frame,
                               text="Save name",
                               command=self.close)
        saveButton.pack()
        self.frame.mainloop()

    def close(self):
        self.name = self.inputtxt.get(1.0, "end-1c")
        self.frame.destroy()


def pickle_dump(obj, file):
    with open(file, "wb") as f:
        pickle.dump(obj, f)


def click_lanes(event, x, y, flags, params):
    global counter, lanes, current_points, lines, finish
    if event == cv2.EVENT_LBUTTONDOWN:
        current_points[counter] = (x, y)
        counter = counter + 1
    if event == cv2.EVENT_RBUTTONDOWN:
        if counter == 0:
            finish = True
        elif counter==2:
            input_box = TextInput()
            name = input_box.name
            new_line = Line(current_points[:counter], name = name)
            lines.append(new_line)
        else:
            input_box = TextInput()
            name = input_box.name
            new_lane = Lane(current_points[:counter], name = name)
            lanes.append(new_lane)
        counter = 0
        current_points = np.zeros((100, 1, 2))




counter = 0
finish = False
lanes, lines = [], []
current_points = np.zeros((100, 1, 2))

def setup_video(inputVideoPath, config_name):
    global counter, lanes, current_points, lines, finish
    videoStream = cv2.VideoCapture(inputVideoPath)
    (grabbed, frame) = videoStream.read()

    while True:
        if finish:
            break
        # Showing original image
        for lane in lanes:
            frame = lane.draw(frame)
        for line in lines:
            line.draw(frame)
        cv2.imshow("Original Image ", frame)
        if cv2.waitKey(20) & 0xFF == 27:
            break

        # Mouse click event on original image
        cv2.setMouseCallback("Original Image ", click_lanes)
        # Printing updated point matrix
        # print(point_matrix)
        # Refreshing window all time
        cv2.waitKey(1)
    cv2.destroyAllWindows()

    pickle_dump([lanes, lines], config_name)
    for obj in lanes + lines:
        print(obj.name)
    return lanes, lines

if __name__ == "__main__":
    setup_video(inputVideoPath = "test_videos/dundrum.mp4",
                config_name = "test_config")