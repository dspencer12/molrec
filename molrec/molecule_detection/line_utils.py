"""
This module defines functions to calculate line properties.

"""
import math


def calculate_segment_length(
        x0: float, y0: float, x1: float, y1: float) -> float:
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
    delta_y = y1 - y0
    delta_x = x1 - x0
    return (delta_y * delta_y + delta_x * delta_x) ** 0.5


def calculate_gradient(
        x0: float, y0: float, x1: float, y1: float) -> float:
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


def calculate_parallel_distance(c1: float, c2: float, m: float) -> float:
    """
    Calculates the distance between two parallel lines based on their
    intercepts and shared gradient.

    c1:

    """
    return abs(c1 - c2) / (m * m + 1) ** 0.5
