import cv2
import numpy as np


def get_corners_from_contour(contour: np.ndarray, corner_number=4) -> None:
    """Gets the corners of a contour.

    cv2 contours found using the cv2.findContours() method are imperfect and not helpful for
    use with tools like solvePNP. This method approximates the contour as a polygon with a specified
    number of corners/sides. It is more accurate than the previous version, as it uses the bisection method,
    rather than just increasing or decreasing a number by a set amount.

    NOTE: The points are in clockwise or anticlockwise depending on the input contour. 
        The returned contour should be checked if the order matters.

    Args:
        contour: a single contour in the format used by opencv.
        corner_number: The number of corners the approximated polygon should have.

    Returns:
        The approximated polygon, in a similar format to the 'contour' parameter.
    """

    arclength = cv2.arcLength(contour, True)
    lower = 0 * arclength
    depth = 15
    upper = 2 * arclength

    for _ in range(depth):
        current = (lower + upper) / 2
        hull = cv2.convexHull(cv2.approxPolyDP(contour, current, True))
        if len(hull) > corner_number:
            lower = current
        else:
            upper = current

    i = 0
    while len(hull) != corner_number and i < 8:
        current = (lower + upper) / 2
        hull = cv2.convexHull(cv2.approxPolyDP(contour, current, True))
        if len(hull) > corner_number:
            lower = current
        else:
            upper = current
        i += 1
    return hull
