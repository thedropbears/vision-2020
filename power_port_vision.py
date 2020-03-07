# pylint: disable=C0330
# pylint: disable=E1101

"""The Drop Bears' 2020 vision code.

This code is run on the Raspberry Pi 4. It is uploaded via the browser interface.
It can be found at https://github.com/thedropbears/vision-2020
"""
import sys
import cv2
import numpy as np
from connection import NTConnection
from camera_manager import CameraManager
from magic_numbers import *
from utilities.functions import *
import math
import time
from typing import Optional, Tuple


class VisionTarget:
    def __init__(self, contour: np.ndarray) -> None:
        """Initialise a vision target object
        
        Args:
            contour: a single numpy/opencv contour
        """
        self.contour = contour.reshape(-1, 2)
        self._test_valid()

    def _test_valid(self):
        self.is_valid_target = True

    def get_leftmost_x(self) -> int:
        return min(list(self.contour[:, 0]))

    def get_rightmost_x(self) -> int:
        return max(list(self.contour[:, 0]))

    def get_middle_x(self) -> int:
        return int((self.get_leftmost_x() + self.get_rightmost_x()) / 2)

    def get_highest_y(self) -> int:
        return min(list(self.contour[:, 1]))

    def get_lowest_y(self) -> int:
        return max(list(self.contour[:, 1]))

    def get_middle_y(self) -> int:
        return int((self.get_highest_y() + self.get_lowest_y()) / 2)


class PowerPort(VisionTarget):
    def _test_valid(self):
        self.contour_area = cv2.contourArea(self.contour)
        if self.contour_area > PP_MIN_CONTOUR_AREA:

            self.convex_hull = cv2.convexHull(self.contour).reshape(-1, 2)
            self.convex_area = cv2.contourArea(self.convex_hull)
            if (
                PP_MAX_AREA_RATIO
                > self.contour_area / self.convex_area
                > PP_MIN_AREA_RATIO
            ):
                self.approximation = get_corners_from_contour(self.contour).reshape(
                    -1, 2
                )
                if len(self.approximation) == 4:
                    self.is_valid_target = True

                else:
                    self.is_valid_target = False
            else:
                print("Failed area ratio check")
                self.is_valid_target = False
        else:
            self.is_valid_target = False


class Vision:
    """Main vision class.

    An instance should be created, with test=False (default). As long as the cameras are configured
    correctly via the GUI interface, everything will work without modification required.
    This will not work on most machines, so tests of the main process function are
    the only tests that can be done without a Pi running the FRC vision image.
    """

    entries = None
    COLOUR_GREEN = (0, 255, 0)
    COLOUR_BLUE = (255, 0, 0)
    COLOUR_RED = (0, 0, 255)
    COLOUR_YELLOW = (0, 255, 255)

    def __init__(self, camera_manager: CameraManager, connection: NTConnection) -> None:
        # self.entries = entries
        # Memory Allocation
        # Numpy takes Rows then Cols as dimensions. Height x Width
        self.hsv = np.zeros(shape=(FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        self.display = self.hsv.copy()
        self.mask = np.zeros(shape=(FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)

        # Camera Configuration
        self.camera_manager = camera_manager

        self.camera_manager.set_camera_property("white_balance_temperature_auto", 0)
        self.camera_manager.set_camera_property("exposure_auto", 1)
        self.camera_manager.set_camera_property("focus_auto", 0)
        self.camera_manager.set_camera_property("exposure_absolute", 1)

        self.connection = connection

        self.old_fps_time = 0

    def find_power_port(self, frame: np.ndarray) -> tuple:
        # Convert to RGB to draw contour on - shouldn't recreate every time
        self.display = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR, dst=self.display)

        _, cnts, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(cnts) >= 1:
            acceptable_cnts = []
            # Check if the found contour is possibly a target
            for current_contour in cnts:
                power_port = PowerPort(current_contour)
                if power_port.is_valid_target:
                    acceptable_cnts.append(power_port)

            if acceptable_cnts:
                if len(acceptable_cnts) > 1:
                    # Pick the largest found 'power port'
                    power_port = max(acceptable_cnts, key=lambda pp: pp.contour_area)
                else:
                    power_port = acceptable_cnts[0]
                return power_port
            else:
                return None
        else:
            return None

    def create_annotated_display(self, frame: np.ndarray, power_port: PowerPort):
        cv2.drawContours(
            frame,
            power_port.approximation.reshape(1, 4, 2),
            -1,
            self.COLOUR_BLUE,
            thickness=2,
        )

        for point in power_port.approximation:
            cv2.circle(frame, tuple(point), 5, self.COLOUR_YELLOW, thickness=2)

        return frame

    def validate_and_reduce_contour(
        self, contour: np.ndarray
    ) -> Optional[Tuple[np.ndarray, int]]:
        """Test if a contour is valid, then returns the contour's area and 4-point reduction.

        Args:
            contour: A single opencv contour
        Returns:
            The contour's area and approximation if it passes the tests, otherwise None

        Tests:
            If the contour area is above a certain amount
            If the ratio of the contour area to convex hull area is within a certain range
            If the contour's approximation has 4 sides
        """
        contour_area = cv2.contourArea(contour)
        if not (contour_area > PP_MIN_CONTOUR_AREA):
            return None

        convex_hull = cv2.convexHull(contour)
        convex_area = cv2.contourArea(convex_hull)
        if not (PP_MAX_AREA_RATIO > contour_area / convex_area > PP_MIN_AREA_RATIO):
            return None

        approximation = get_corners_from_contour(contour)
        if not (len(approximation) == 4):
            return None

        return contour_area, approximation

    def get_mid(self, contour: np.ndarray) -> tuple:
        """ Use the cv2 moments to find the centre x of the contour.
        We just copied it from the opencv reference. The y is just the lowest
        pixel in the image."""
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
        else:
            cX = 160
        return cX

    def get_image_values(self, frame: np.ndarray) -> tuple:
        """Takes a frame, returns a tuple of results, or None."""
        self.hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV, dst=self.hsv)
        self.mask = cv2.inRange(
            self.hsv, HSV_LOWER_BOUND, HSV_UPPER_BOUND, dst=self.mask
        )

        power_port = self.find_power_port(self.mask)

        if power_port is not None:
            self.display = self.create_annotated_display(self.display, power_port)
            midX = power_port.get_middle_x()

            target_top = power_port.get_highest_y()
            # target_bottom = max(list(power_port[:, :, 1]))
            # print("target top: ", target_top, " target bottom: ", target_bottom)
            horiz_angle = get_horizontal_angle(
                midX, FRAME_WIDTH, MAX_FOV_WIDTH / 2, True
            )

            vert_angle = get_vertical_angle_linear(
                target_top, FRAME_HEIGHT, MAX_FOV_HEIGHT / 2, True
            )

            distance = get_distance(
                vert_angle, TARGET_HEIGHT_TOP, CAMERA_HEIGHT, GROUND_ANGLE
            )
            print("angle: ", math.degrees(vert_angle), " distance: ", distance)

            return (distance, horiz_angle)
        else:
            return None

    def run(self):

        """Main process function.
        When ran, takes image, processes image, and sends results to RIO.
        """
        self.connection.pong()

        frame_time, self.frame = self.camera_manager.get_frame()
        if frame_time == 0:
            self.camera_manager.notify_error(self.camera_manager.get_error())
            return
        # Flip the image cause originally upside down.
        self.frame = cv2.rotate(self.frame, cv2.ROTATE_180)
        results = self.get_image_values(self.frame)

        self.connection.set_fps()

        if results is not None:
            distance, angle = results
            self.connection.send_results(
                (distance, angle, time.monotonic())
            )  # distance (meters), angle (radians), timestamp
        self.camera_manager.send_frame(self.display)


if __name__ == "__main__":
    vision = Vision(
        CameraManager("Power Port Camera", "/dev/video0", 240, 320, 30, "kYUYV"),
        NTConnection(),
    )
    while True:
        vision.run()
