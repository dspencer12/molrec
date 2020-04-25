"""

"""
from typing import Optional, Tuple

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
        rgb_colour: Flat image colour, defaults to white.

    Returns:
        Image array

    """
    image = np.zeros((width, height, 3), np.uint8)
    # OpenCV using BGR colours, so RGB should be reversed
    image[:] = tuple(reversed(rgb_colour or (255, 255, 255)))
    return image
