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


# The next two functions' `x` and `y` arguments are in camera coordinates.
# The origin is the top left corner of the screen, with left and down being positive.

# Both have an `invert` argument which, if False, leaves the return angle with negative
# at the origin, and if True, has positive at the origin.

# The intrinsic matrix should look like this:
# np.array([
#    [FX, 0.0, CX],
#    [0.0, FY, CY],
#    [0.0, 0.0, 1.0],
# ])
# Where FX and FY are the focal lengths in the horizontal and vertical
# sizes of pixels, respectively, and CX and CY are the centres of the frame.
# These values should be calculated by calibration, but can be calculated manually.


def get_vertical_angle(y: int, intr_matrix: np.ndarray, inverted=False) -> float:
    """Get the vertical angle of a point to the camera's centre.

    Args:
        y: The `y` of a point in camera coordinates. (As explained above)
        intr_matrix: The camera's intrinsic properties (As explained above)
        inverted: Inverts the output angle (As explained above)
    Returns:
        An angle, in radians.
    """
    if inverted:
        return math.atan2(intr_matrix[1][2] - y, intr_matrix[1][1])
    else:
        return math.atan2(y - intr_matrix[1][2], intr_matrix[1][1])


def get_horizontal_angle(x: int, intr_matrix: np.ndarray, inverted=False) -> float:
    """Get the horizontal angle of a point to the camera's centre.

    Args:
        y: The `x` of a point in camera coordinates. (As explained above)
        intr_matrix: The camera's intrinsic properties (As explained above)
        inverted: Inverts the output angle (As explained above)
    Returns:
        An angle, in radians.
    """
    if inverted:
        return math.atan2(intr_matrix[0][2] - x, intr_matrix[0][0])
    else:
        return math.atan2(x - intr_matrix[0][2], intr_matrix[0][0])


def get_distance(
    target_angle: float, target_height: float, camera_height: float, camera_tilt: float
) -> None:
    """Gets the ground distance from the camera to the target.

    Args:
        target_angle: The angle of the point below the centre plane of the camera. 
            Downwards is positive, as returned by get_vertical_angle()'s default.
        target_height: The height of the target above the ground (in metres)
        camera_height: The height of the camera above the ground (in metres)
        camera_tilt: The camera's angle of elevation from the ground (upwards is positive)
    Returns:
        A positive perpendicular distance to the target along the ground (in metres)
    """
    return (target_height - camera_height) / math.tan(camera_tilt - target_angle)
