from my_utils import *

def parse_outputs(layerOutputs, video_width, video_height):
    boxes, confidences, classids = [], [], []
    for output in layerOutputs:
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
                classids.append(classID)
    return boxes, confidences, classids


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




def get_most_recent_position(previous_frame_detections):
    most_recent_positions = {}
    for detections in previous_frame_detections:
        prev_labels = {label:center for center, label in detections.items()}
        for label in prev_labels.keys():
            most_recent_positions[label] = prev_labels[label]
    most_recent_positions = {label:center for center, label in most_recent_positions.items()}
    return most_recent_positions


def identify_vehicles(boxes,
                      confidences,
                      classids,
                      frame,
                      vehicle_count,
                      all_detections):
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, preDefinedConfidence,
                            preDefinedThreshold)
    drawDetectionBoxes(idxs, boxes, classids, confidences, frame)
    previous_detections = all_detections[-FRAMES_BEFORE_CURRENT:]
    vehicle_count, current_detections = count_vehicles(idxs,
                                                       boxes,
                                                       classids,
                                                       vehicle_count,
                                                       previous_detections,
                                                       frame)
    current_positions = {label: center for center, label in current_detections.items()}
    return idxs, vehicle_count, current_detections, current_positions