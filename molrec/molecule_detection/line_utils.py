def calculate_segment_length(x0, y0, x1, y1):
    return ((y1 - y0) ** 2 + (x1 - x0) ** 2) ** 0.5


def calculate_gradient(x0, y0, x1, y1):
    delta_x = x1 - x0
    return (y1 - y0) / delta_x if delta_x > 0. else 0.


def calculate_intercept(x: float, y: float, m: float) -> float:
    """
    Calculates the y-intercept.

    Args:

    """
    return y - m * x


def calculate_parallel_distance(c1: float, c2: float, m: float) -> float:
    """
    Calculates the distance between two parallel lines based on their
    intercepts and shared gradient.

    """
    return abs(c1 - c2) / (m * m + 1) ** 0.5
