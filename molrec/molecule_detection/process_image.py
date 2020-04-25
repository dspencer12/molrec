import math
from typing import Optional, Tuple

import cv2
import numpy as np

from . import (
    feature_detection,
    line_utils
)


def process_molecule_image(filename: str)\
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """

    :param filename:
    :return:
    """
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    corners = feature_detection.get_corners(gray)

    lines = feature_detection.get_edges(gray)

    lines = remove_parallel_edges(
        lines,
        gradient_tolerance=0.1,
        min_line_dist=20.
    )

    return img, corners, lines


def remove_parallel_edges(
        edges,
        gradient_tolerance: float = 0.1,
        min_line_dist: float = 10.
) -> np.ndarray:
    """

    :param edges:
    :param gradient_tolerance:
    :param min_line_dist:
    :return:
    """
    coords = [edge[0] for edge in edges]

    # TODO: vectorize these computations
    lengths = [line_utils.calculate_segment_length(*a.T) for a in coords]
    gradients = [line_utils.calculate_gradient(*a.T) for a in coords]
    intercepts = [
        line_utils.calculate_intercept(a[0], a[1], gradients[ii])
        for ii, a in enumerate(coords)
    ]

    keep = [True] * len(coords)
    for ii, edge1 in enumerate(coords):
        if not keep[ii]:
            continue
        for jj, edge2 in enumerate(coords[ii + 1:], ii + 1):
            if not keep[jj]:
                continue

            grad = gradients[ii]

            dist: Optional[float] = None
            if math.isclose(grad, gradients[jj], abs_tol=gradient_tolerance):
                dist = line_utils.calculate_parallel_distance(
                    intercepts[ii], intercepts[jj], grad)
            elif math.isnan(grad) and math.isnan(gradients[jj]):
                # Vertical lines
                dist = abs(edge1[0] - edge2[0])

            if dist is not None and dist <= min_line_dist:
                # Keep longest segment
                idx = ii if lengths[ii] < lengths[jj] else jj
                keep[idx] = False

    return edges[keep]
