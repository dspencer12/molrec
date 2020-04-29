"""

"""
from typing import Optional, Tuple, Union

import numpy as np


PixelCoord = Tuple[int, int]
RGBColour = Tuple[int, int, int]


def create_blank_image(
        width: int,
        height: int,
        rgb_colour: Optional[RGBColour] = None
) -> np.ndarray:
    """
    Initializes an empty image array of `width` by `height`.

    Args:
        width: Desired image width.
        height: Desired image height.
        rgb_colour: Flat image colour, defaults to white. Note that the default
                    is enacted in the function body to enable pass-through of
                    None values.

    Returns:
        Image array

    """
    image = np.zeros((width, height, 3), np.uint8)
    # OpenCV uses BGR colours, so RGB should be reversed
    image[:] = tuple(reversed(rgb_colour or (255, 255, 255)))
    return image


def get_average_point(coords: np.ndarray) -> np.ndarray:
    """
    Calculates the average point of the `coords`, AKA the centre of mass.

    Args:
        coords: An N x 2 array of coordinates.

    Returns:
        1 x 2 array corresponding to the (x, y) coordinates of the centre,
        rounded to integers.

    """
    return np.around(np.mean(coords, axis=0)).astype(np.int64)


def rotate_coordinates(
        coords: np.ndarray,
        angle: float,
        point: Union[PixelCoord, np.ndarray] = (0, 0)
) -> np.ndarray:
    """
    Rotates the `coords` around point `point` by the given `angle`.

    Args:
        coords: An N x 2 array of coordinates.
        angle: Rotation angle in radians.
        point: The point around which to rotate. Defaults to the origin.

    Returns:
        Transformed coordinates as an N x 2 array.

    """
    angle_cos, angle_sin = np.cos(angle), np.sin(angle)
    rot_mat = np.array([[angle_cos, angle_sin], [-angle_sin, angle_cos]])
    transformed = np.dot(rot_mat, (coords - point).T).T + point
    return np.around(transformed).astype(np.int64)
