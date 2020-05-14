import math
from typing import List, Optional, Tuple

import cv2
import numpy as np

from . import line_utils


class DetectionError(Exception):
    """General exception for problems during feature detection."""


def detect_corners(
        image: np.ndarray,
        max_corners: int = 1000,
        quality_level: float = 0.3,
        min_distance: int = 10
) -> np.ndarray:
    """
    Detects vertices in the given `image`.
    """
    corners = cv2.goodFeaturesToTrack(
        image,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance
    )
    if corners is None:
        raise DetectionError('No vertices detected in image.')
    return np.around(corners).astype('int64')


def test_coords_close(
        a: Tuple[int, int],
        b: Tuple[int, int],
        tol: int
) -> bool:
    """
    Determines whether coordinates `a` and `b` lie within `tol` of one another.

    This test is based on the Cartesian distance between the two coordinates.

    """
    return line_utils.calculate_point_distance(a, b) <= tol


def get_vertices_from_edges(
        edges: np.ndarray,
        image_size: Tuple[int, int],
        tolerance: Optional[int] = None
):
    """
    Identifies unique vertices based on the `edges`.

    If `tolerance` is not provided, it is defined adaptively based on
    `image_size`.

    """
    if tolerance is None:
        tolerance = image_size[0] // 50

    vertices: List[Tuple[int, int]] = []
    for edge in edges:
        # TODO: Should the average vertex coordinate be kept, rather than the
        #  first encountered?
        for vertex in [tuple(edge[0, :2]), tuple(edge[0, 2:])]:
            if not any(test_coords_close(vertex, v, tolerance)
                       for v in vertices):
                vertices.append(vertex)
    # This return format mimics the corner detection format from
    # cv2.goodFeaturesToTrack
    return np.array([[v] for v in vertices])


def remove_parallel_edges(
        edges: np.ndarray,
        gradient_tolerance: float = 0.1,
        max_line_dist: float = 20.
) -> np.ndarray:
    """
    Removes parallel lines in close proximity to one another from `edges`.

    Args:
        edges: Array of line coordinates (start and end point of each line).
        gradient_tolerance: Maximum allowed difference in gradient for lines to
                            be considered parallel. Defaults to 0.1.
        max_line_dist: Maximum distance between lines for them to be considered
                       adjacent to one another.

    Returns:
        Array with adjacent parallel lines removed.

    """
    coords = [edge[0] for edge in edges]

    # TODO: vectorise these computations
    lengths = [line_utils.calculate_segment_length(*a.T) for a in coords]
    gradients = [line_utils.calculate_gradient(*a.T) for a in coords]

    keep = [True] * len(coords)
    for ii, edge1 in enumerate(coords):
        if not keep[ii]:
            continue
        for jj, edge2 in enumerate(coords[ii + 1:], ii + 1):
            if not keep[jj]:
                continue

            grad = gradients[ii]

            if (math.isclose(grad, gradients[jj], abs_tol=gradient_tolerance) or
                    (math.isnan(grad) and math.isnan(gradients[jj]))):
                if math.isclose(grad, gradients[jj], abs_tol=0.00000001):
                    # "Identical" gradients - skip unnecessary calculations
                    dist = line_utils.calculate_point_segment_distance(
                        tuple(edge1[:2]),
                        tuple(edge1[2:]),
                        (edge2[0], edge2[1])
                    )
                    if dist <= max_line_dist:
                        # Keep longest segment
                        idx = ii if lengths[ii] < lengths[jj] else jj
                        keep[idx] = False
                else:
                    # Test each segment endpoint against the other segment
                    combinations = [
                        (edge1, tuple(edge2[:2])),
                        (edge1, tuple(edge2[2:])),
                        (edge2, tuple(edge1[:2])),
                        (edge2, tuple(edge1[2:]))
                    ]
                    for edge, point in combinations:
                        dist = line_utils.calculate_point_segment_distance(
                            tuple(edge[:2]),
                            tuple(edge[2:]),
                            point
                        )
                        if dist <= max_line_dist:
                            # Keep longest segment
                            idx = ii if lengths[ii] < lengths[jj] else jj
                            keep[idx] = False
                            break

    return edges[keep]


def detect_edges(
    image: np.ndarray,
    remove_parallel: bool = True,
    max_line_dist: Optional[float] = None
) -> np.ndarray:
    """
    Detects edges (lines) in the given `image` using the probabilistic
    Hough line transform.

    Args:
        image: The image array.
        remove_parallel: Flag indicating whether nearby parallel edges should be
                         filtered.
        max_line_dist: Maximum distance between lines for them to be considered
                       adjacent to one another. Defaults to the x-size of the
                       image divided by 200.

    """
    image = image.astype(np.uint8)

    detector = cv2.ximgproc.createFastLineDetector(
        _canny_aperture_size=7
    )
    lines = detector.detect(image)

    if lines is None:
        raise DetectionError('No edges found in image.')

    lines = np.around(lines).astype('int64')

    if remove_parallel:
        if max_line_dist is None:
            max_line_dist = image.shape[0] / 200
        lines = remove_parallel_edges(lines, max_line_dist=max_line_dist)

    return lines
