from my_utils import *
from tracking import *


def get_pytorch_outputs(model, frames):
    with torch.no_grad():
        results = model(frames)
    results_dfs = results.pandas().xyxy
    # results_df = results.pandas().xyxy[0]
    boxes, confidences, classids = [], [], []
    for frame, output_df in zip(frames, results_dfs):
        output_df["width"] = (output_df.xmax - output_df.xmin).astype("int")
        output_df["height"] = (output_df.ymax - output_df.ymin).astype("int")
        x = list(output_df.xmin.astype("int"))
        y = list(output_df.ymin.astype("int"))
        width = list(output_df.width)
        height = list(output_df.height)
        boxes.append(list(zip(x, y, width, height)))
        classids.append(list(output_df['class']))
        confidences.append(list(output_df.confidence))
    return boxes, confidences, classids


def get_cv2_outputs(model, frames):
    frame = frames[0]
    output_layers = []
    blob = cv2.dnn.blobFromImage(frame,
                                 1 / 255.0,
                                 (inputWidth, inputHeight),
                                 swapRB=True, crop=False)
    model.setInput(blob)
    outputs = model.forward(output_layers)
    boxes, confidences, classids = parse_outputs(outputs,
                                                 frame.shape[1],
                                                 frame.shape[0])

    return [boxes], [confidences], [classids]




if MODEL_TYPE == "cv2":
    BATCH_SIZE = 1
    print("[INFO] loading YOLO from disk...")
    model = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    output_layers = model.getLayerNames()
    output_layers = [output_layers[i[0] - 1] for i in model.getUnconnectedOutLayers()]

    get_model_output = get_cv2_outputs
elif MODEL_TYPE == "pytorch":
    import torch
    model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)
    # model = model.cpu()
    get_model_output = get_pytorch_outputs
else:
    raise ValueError("MODEL_TYPE must be cv2 or pytorch")

