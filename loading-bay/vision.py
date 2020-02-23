from camera_manager import CameraManager
from magic_numbers import *
from utilities.functions import (
    get_corners_from_contour,
    get_values_solvepnp,
    order_rectangle,
)
import cv2
import numpy as np
import time
import math


class Vision:
    def __init__(self, camera_manager: CameraManager) -> None:
        self.camera_manager = camera_manager

        self.image = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), np.uint8)
        self.mask = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), np.uint8)
        self.hsv = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), np.uint8)

    def run(self) -> None:
        """Run the main vision loop once."""
        frame_time, self.frame = self.camera_manager.get_frame()
        if not frame_time:
            error = self.camera_manager.get_error()
            self.camera_manager.notify_error(error)
            return

        self.frame = cv2.rotate(self.frame, cv2.ROTATE_180, dst=self.frame)
        results = self.get_image_values(self.frame)

        if results is not None:
            distance, angle = results
            self.connection.send_results((distance, angle, time.monotonic()))

        self.camera_manager.send_frame(self.image)

    def get_image_values(self, frame: np.ndarray):
        self.image = self.frame
        return None

    def test_contour_pair(self, inner: dict, outer: dict) -> bool:
        """Test if a given contour pair is a valid loading bay.

        Tests include (in order):
            The number of points in the inner contour is greater than 3,
            The number of points in the outer contour is greater than 3,
            The ratio of the areas of the two contours,
            The approximated inner contour has 4 sides,
            The approximated outer contour has 4 sides,
            The ratio of the areas of the inner contour and its approximated contour,
            The ratio of the areas of the outer contour and its approximated contour,

        Args:
            inner: The inner contour (a dictionary (like how was created earlier)).
            outer: The outer contour (a dictionary (like how was created earlier))
        Returns:
            A boolean of whether or not the pair is valid.
            inner: The inner contour (with the rectangle added)
            outer: The outer contour (with the rectangle and area added)
        """

        if not (inner["contour"].shape[0] > 3):
            return False, inner, outer
        if not (outer["contour"].shape[0] > 3):
            return False, inner, outer

        outer["area"] = cv2.contourArea(outer["contour"])

        if not (
            abs(outer["area"] / inner["area"] - INNER_OUTER_RATIO) < INNER_OUTER_ERROR
        ):
            return False, inner, outer

        inner["rect"] = get_corners_from_contour(inner["contour"])

        if not len(inner["rect"] == 4):
            return False, inner, outer

        outer["rect"] = get_corners_from_contour(outer["contour"])

        if not len(outer["rect"] == 4):
            return False, inner, outer

        if not (
            abs(inner["area"] / cv2.contourArea(inner["rect"]) - 1)
            < RAW_RECT_AREA_ERROR
        ):
            return False, inner, outer
        if not (
            abs(outer["area"] / cv2.contourArea(outer["rect"]) - 1)
            < RAW_RECT_AREA_ERROR
        ):
            return False, inner, outer
        return True, inner, outer

    def draw_contour_pair(self, inner: dict, outer: dict) -> None:
        self.image = cv2.drawContours(
            self.image, inner["rect"].reshape((1, 4, 2)), -1, (255, 0, 0), thickness=2
        )
        self.image = cv2.drawContours(
            self.image, outer["rect"].reshape((1, 4, 2)), -1, (0, 0, 255), thickness=2
        )
