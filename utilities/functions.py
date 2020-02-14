import cv2
import numpy as np
import math


def scale_value(
    value: float,
    input_lower: float,
    input_upper: float,
    output_lower: float,
    output_upper: float,
    exponent: float = 1,
) -> float:
    """Scales a value based on the input range and output range.
    For example, to scale a joystick throttle (1 to -1) to 0-1, we would:
        scale_value(joystick.getThrottle(), 1, -1, 0, 1)
    The output is then raised to the exponent argument.

    Args:
        value: The value to be scaled.
        input_lower: The `lower` end of the input range. Note that lower in this case
            does not necessarily mean lower, just that it maps to the `lower` output.
        input_upper: Similar to `input_lower`, but the other bound.
        output_lower: The value which `input_lower` should scale to.
        output_upper: The value which `input_upper` should scale to.
        exponent (optional): The power to raise the result to.
            Note that sign is preserved.
    Returns:
        The scaled value (es explained above).
    """

    input_distance = input_upper - input_lower
    output_distance = output_upper - output_lower
    ratio = (value - input_lower) / input_distance
    result = ratio * output_distance + output_lower
    return math.copysign(result ** exponent, result)


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
    depth = 8
    upper = 2 * arclength

    # The cv2.arcLength function gets the perimiter of a contour.
    # The approxPolyDP function's second argument is a 'tolerance'.
    # Sometimes when the tolerance is within a certain range,
    # it will return a contour with the right number of sides,
    # but not a good approximation. So we use the bisection method
    # to get the lowest tolerance that still has the right number of
    # sides. That is why we do it in two parts: one to get the required
    # accuracy, and then another to ensure the right amount of sides.

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
