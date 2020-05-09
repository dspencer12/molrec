"""
This module provides utility functions for working with images.

"""

import cv2


def annotate_image(image, corners=None, lines=None):
    """
    Annotates the provided `image` using circles for `corners` and lines for
    `lines`.

    """
    if corners is not None:
        for i in corners:
            x, y = i.ravel()
            cv2.circle(image, (x, y), 10, 255, -1)

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return image
