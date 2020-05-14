import os

import cv2
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def east_detection(
        image: np.ndarray,
        min_confidence: float = 0.5,
        apply_suppression: bool = True
) -> np.ndarray:
    """
    Performs EAST text detection of text bounding boxes.

    Based on the example at
    https://www.pyimagesearch.com/2018/08/20/
    opencv-text-detection-east-text-detector/

    Args:
        image: Numpy array image. Note that the dimensions must be multiples of
               32.
        min_confidence: The minimum probability required for a bounding box.
        apply_suppression: Whether to perform non-maximal suppression.

    Returns:
        Numpy array containing the coordinates of the start and end points of
        each bounding box.

    """
    # Define the two output layer names for the EAST detector model in which
    # we are interested - the first is the output probabilities and the
    # second can be used to derive the bounding box coordinates of text
    layer_names = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"
    ]

    net = cv2.dnn.readNet(
        os.path.join(SCRIPT_DIR, 'frozen_east_text_detection.pb')
    )

    blob = cv2.dnn.blobFromImage(
        image,
        1.0,
        tuple(image.shape[:2]),
        (123.68, 116.78, 103.94),
        swapRB=True,
        crop=False
    )
    net.setInput(blob)
    scores, geometry = net.forward(layer_names)

    num_rows, num_cols = scores.shape[2:4]
    rects = []
    confidences = []
    for y in range(0, num_rows):
        # Extract the scores (probabilities), followed by the geometrical
        # data used to derive potential bounding box coordinates that
        # surround text
        scores_data = scores[0, 0, y]
        x_data0 = geometry[0, 0, y]
        x_data1 = geometry[0, 1, y]
        x_data2 = geometry[0, 2, y]
        x_data3 = geometry[0, 3, y]
        angles_data = geometry[0, 4, y]

        for x in range(0, num_cols):
            # If our score does not have sufficient probability, ignore it
            if scores_data[x] < min_confidence:
                continue
            # Compute the offset factor as our resulting feature maps will
            # be 4x smaller than the input image
            offset_x, offset_y = (x * 4.0, y * 4.0)
            # Extract the rotation angle for the prediction
            angle = angles_data[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            # Use the geometry volume to derive the width and height of
            # the bounding box
            h = x_data0[x] + x_data2[x]
            w = x_data1[x] + x_data3[x]
            # Compute both the starting and ending (x, y)-coordinates for
            # the text prediction bounding box
            end_x = int(offset_x + (cos * x_data1[x]) + (sin * x_data2[x]))
            end_y = int(offset_y - (sin * x_data1[x]) + (cos * x_data2[x]))
            start_x = int(end_x - w)
            start_y = int(end_y - h)

            rects.append((start_x, start_y, end_x, end_y))
            confidences.append(scores_data[x])

    if apply_suppression and rects:
        # Apply non-maximal suppression to suppress weak, overlapping bounding
        # boxes
        # Note that NMSBoxes is very fussy about its inputs:
        # https://github.com/opencv/opencv/issues/12299
        indices = cv2.dnn.NMSBoxes(
            [list(r) for r in rects],
            [float(c) for c in confidences],
            min_confidence,
            # I have no idea how this threshold works... higher appears to
            # retain more boxes
            nms_threshold=0.4
        )
        rects = np.array(rects)[indices[:, 0]]
    else:
        rects = np.array(rects)

    return rects
