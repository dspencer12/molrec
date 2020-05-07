"""
This module defines functions to calculate line properties.

"""
import math
from typing import Tuple

import numpy as np


def calculate_segment_length_squared(
        x0: float, y0: float, x1: float, y1: float
) -> float:
    """
    Calculates the square of the length of a line segment, given the start and
    end points of the segment.

    Args:
        x0: x-coordinate of the start point of the line segment.
        y0: y-coordinate of the start point of the line segment.
        x1: x-coordinate of the end point of the line segment.
        y1: y-coordinate of the end point of the line segment.

    Returns:
        Squared length of the line segment.

    """
    delta_y = y1 - y0
    delta_x = x1 - x0
    return delta_y * delta_y + delta_x * delta_x


def calculate_segment_length(
        x0: float, y0: float, x1: float, y1: float
) -> float:
    """
    Calculates the length of a line segment, given the start and end points
    of the segment.

    Args:
        x0: x-coordinate of the start point of the line segment.
        y0: y-coordinate of the start point of the line segment.
        x1: x-coordinate of the end point of the line segment.
        y1: y-coordinate of the end point of the line segment.

    Returns:
        Length of the line segment.

    """
    return calculate_segment_length_squared(x0, y0, x1, y1) ** 0.5


def calculate_gradient(
        x0: float, y0: float, x1: float, y1: float
) -> float:
    """
    Calculates the gradient of a line, given two points on that line.

    Args:
        x0: x-coordinate of the first point on the line.
        y0: y-coordinate of the first point on the line.
        x1: x-coordinate of the second point on the line.
        y1: y-coordinate of the second point on the line.

    Returns:
        Length of the line segment. If the line is vertical, this is NaN.
    """
    delta_x = x1 - x0
    if delta_x == 0.:
        return math.nan
    return (y1 - y0) / (x1 - x0)


def calculate_intercept(x: float, y: float, m: float) -> float:
    """
    Calculates the y-intercept given a point on the line and the gradient.

    If the gradient, `m`, of the line is NaN, then the intercept will similarly
    be NaN.

    Args:
        x: x-coordinate of a point on the line.
        y: y-coordinate of a point on the line.
        m: gradient of the line.

    Returns:
        The y-intercept of the line.

    """
    return y - m * x


def calculate_midpoint(
        x0: float, y0: float, x1: float, y1: float
) -> Tuple[float, float]:
    """
    Calculates the midpoint of a line segment.

    Args:
        x0: x-coordinate of the first point on the line.
        y0: y-coordinate of the first point on the line.
        x1: x-coordinate of the second point on the line.
        y1: y-coordinate of the second point on the line.

    Returns:
        Midpoint of the line segment.

    """
    return (x0 + x1) / 2, (y0 + y1) / 2


def calculate_point_segment_distance(
        start: Tuple[float, float],
        end: Tuple[float, float],
        point: Tuple[float, float]
) -> float:
    """
    Calculates the distance between a point `point` and a line segment, defined
    by its two endpoints.

    This implementation is based on the accepted answer to
    https://stackoverflow.com/questions/849211.

    Args:
        start: The start point of the segment as a two-tuple.
        end: The end point of the segment as a two-tuple.
        point: The point from which to measure as a two-tuple.

    Returns:
        Shortest distance between `point` and the specified line segment.

    """
    squared_length = calculate_segment_length_squared(*start, *end)

    if squared_length == 0.:
        # Start point == end point
        return calculate_point_distance(point, start)

    # Consider the line extending the segment, parameterized as
    # start + t(end - start). We find the projection of the point onto the line.
    # It falls where t = [(point - start).(end - start)] / | end - start | ^ 2
    # We clamp t from [0, 1] to handle points outside the line segment.

    start_arr = np.array(start)
    end_arr = np.array(end)
    point_arr = np.array(point)
    t = max(
        0,
        min(
            1,
            np.dot(point_arr - start_arr, end_arr - start_arr) / squared_length
        )
    )

    # Projection falls on the segment
    projection = start_arr + t * (end_arr - start_arr)

    return calculate_point_distance(point, projection)


def calculate_point_distance(
        a: Tuple[float, float],
        b: Tuple[float, float]
) -> float:
    """
    Calculates the distance between point `a` and point `b`.

    Args:
        a: A two-tuple coordinate.
        b: A two-tuple coordinate.

    Returns distance between `a` and `b`.

    """
    return calculate_segment_length(*a, *b)


def calculate_parallel_distance(c1: float, c2: float, m: float) -> float:
    """
    Calculates the distance between two parallel lines based on their
    intercepts and shared gradient.

    c1:

    """
    return abs(c1 - c2) / (m * m + 1) ** 0.5
