import cv2
import numpy as np


def get_corners(img) -> np.ndarray:
    """

    :param img:
    :return:
    """
    corners = cv2.goodFeaturesToTrack(img, 25, 0.01, 10)
    return np.int0(corners)


def get_edges(img) -> np.ndarray:
    """

    :param img:
    :return:
    """
    img = img.astype(np.uint8)
    mu, sigma = cv2.meanStdDev(img)
    edges = cv2.Canny(img, mu - sigma, mu + sigma)
    return cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=100,
        minLineLength=100,
        maxLineGap=100
    )
