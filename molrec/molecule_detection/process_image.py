from typing import Tuple

import cv2
import numpy as np

from . import feature_detection


def process_molecule_image(filename: str)\
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    """
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    corners = feature_detection.get_corners(gray)

    lines = feature_detection.get_edges(gray, remove_parallel=True)

    return img, corners, lines
