from typing import Tuple, Union

import cv2
import numpy as np


def create_mask(
        shape: Tuple[int, ...],
        mask_rectangles: np.ndarray
) -> np.ndarray:
    """
    Constructs an image mask with masked rectangular regions.

    The returned mask will have a white background, with the `mask_rectangles`
    regions in black.

    Args:
        shape: The shape of the mask.
        mask_rectangles: The rectangular regions to mask, defined by start and
                         end coordinates.

    Returns:
        Numpy array mask.

    """
    mask = np.zeros(shape, dtype=np.uint8)
    mask.fill(255)

    mask_colour = tuple([0] * len(shape))

    for (start_x, start_y, end_x, end_y) in mask_rectangles:
        cv2.rectangle(mask, (start_x, start_y), (end_x, end_y), mask_colour, -1)

    return mask


def apply_mask(
        image: np.ndarray,
        mask: np.ndarray,
        apply_colour: Union[int, Tuple[int, int, int]]
) -> np.ndarray:
    """
    Makes a copy of `image` and applies `mask` to the copy.

    Args:
        image: The numpy array image.
        mask: The mask to apply to `image`.
        apply_colour: The desired colour of the masked region.

    """
    masked_image = image.copy()
    if len(image.shape) == 2:
        masked_image[mask == 0] = apply_colour
    else:
        masked_image[np.all(mask == (0, 0, 0), axis=-1)] = apply_colour
    return masked_image
