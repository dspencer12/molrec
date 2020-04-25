import cv2
import numpy as np


def get_corners(image: np.ndarray) -> np.ndarray:
    """
    Detects vertices in the given `image`.
    """
    corners = cv2.goodFeaturesToTrack(image, 25, 0.01, 10)
    return np.int0(corners)


def get_edges(image) -> np.ndarray:
    """
    Detects edges (lines) in the given `image` using the probabilistic
    Hough line transform.

    """
    image = image.astype(np.uint8)
    mu, sigma = cv2.meanStdDev(image)
    edges = cv2.Canny(image, mu - sigma, mu + sigma)
    return cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=100,
        minLineLength=100,
        maxLineGap=100
    )
