from camera_manager import CameraManager
from connection import NTConnection
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
    def __init__(self, camera_manager: CameraManager, connection: NTConnection) -> None:
        self.camera_manager = camera_manager
        self.connection = connection

        self.image = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), np.uint8)
        self.mask = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), np.uint8)
        self.hsv = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), np.uint8)

    def run(self) -> None:
        """Run the main vision loop once."""
        self.connection.pong()
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
        self.hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV, dst=self.hsv)
        self.mask = cv2.inRange(
            self.hsv,
            LOADING_BAY_HSV_LOWER_BOUND,
            LOADING_BAY_HSV_UPPER_BOUND,
            dst=self.mask,
        )
        self.image = cv2.cvtColor(self.mask, cv2.COLOR_GRAY2BGR, dst=self.image)

        _, cnts, hierarchy = cv2.findContours(
            self.mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )

        cnts = np.array(cnts)

        if not len(cnts):
            return None

        hierarchy = np.array(hierarchy[0])
        inner_rects = {}
        outer_rects = {}

        for i, cnt in enumerate(cnts):
            # The 3rd (4th) element of hierarchy is its parent.
            # When parent is -1, it has no parent and therefore it an outer rectangle.
            if hierarchy[i][3] == -1:
                outer_rects[i] = {"contour": cnt, "hierarchy": hierarchy[i]}
            else:
                inner_rects[i] = {"contour": cnt, "hierarchy": hierarchy[i]}

        if not (inner_rects and outer_rects):
            return None

        good_pairs = []
        for i in inner_rects:
            inner_rects[i]["area"] = cv2.contourArea(inner_rects[i]["contour"])
            if inner_rects[i]["area"] > MIN_CONTOUR_AREA:
                parent = outer_rects[inner_rects[i]["hierarchy"][3]]
                success, inner, outer = self.test_contour_pair(inner_rects[i], parent)
                if success:
                    good_pairs.append((inner_rects[i], parent))

        if not good_pairs:
            return None

        for inner, outer in good_pairs:
            self.draw_contour_pair(inner, outer)

        ordered_points = np.concatenate(
            (
                order_rectangle(outer["rect"], inverted=True),
                order_rectangle(inner["rect"], inverted=True),
            )
        )
        results = get_values_solvepnp(
            ordered_points.astype(np.float32),
            LOADING_BAY_POINTS,
            C920_2_INTR_MATRIX,
            C920_2_DIST_COEFFS,
        )
        print(results)

        return results

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
            abs(outer["area"] / inner["area"] - LOADING_INNER_OUTER_RATIO)
            < INNER_OUTER_ERROR
        ):
            return False, inner, outer

        inner["rect"] = get_corners_from_contour(inner["contour"])

        if not len(inner["rect"]) == 4:
            return False, inner, outer

        outer["rect"] = get_corners_from_contour(outer["contour"])

        if not len(outer["rect"]) == 4:
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
        """Draw the inner and outer contours on self.image"""
        self.image = cv2.drawContours(
            self.image, inner["rect"].reshape((1, 4, 2)), -1, (255, 0, 0), thickness=2
        )
        self.image = cv2.drawContours(
            self.image, outer["rect"].reshape((1, 4, 2)), -1, (0, 0, 255), thickness=2
        )


if __name__ == "__main__":
    vision = Vision(
        CameraManager(
            [
                {
                    "name": "Power Port Camera",
                    "path": "/dev/video0",
                    "config": {
                        "pixel format": "yuyv",
                        "fps": 30,
                        "height": 240,
                        "width": 320,
                        "stream": {"properties": []},
                    },
                }
            ]
        ),
        NTConnection(),
    )
    while True:
        vision.run()
