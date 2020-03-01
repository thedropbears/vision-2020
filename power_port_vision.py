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

    def __init__(self, camera_manager: CameraManager, connection: NTConnection) -> None:
        # self.entries = entries
        # Memory Allocation
        # Numpy takes Rows then Cols as dimensions. Height x Width
        self.hsv = np.zeros(shape=(FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        self.display = self.hsv.copy()
        self.mask = np.zeros(shape=(FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)

        # Camera Configuration
        self.camera_manager = camera_manager

        self.camera_manager.set_camera_property(0, "white_balance_temperature_auto", 0)
        self.camera_manager.set_camera_property(0, "exposure_auto", 1)
        self.camera_manager.set_camera_property(0, "focus_auto", 0)
        self.camera_manager.set_camera_property(0, "exposure_absolute", 1)

        self.connection = connection

        self.old_fps_time = 0

    def find_power_port(self, frame: np.ndarray) -> tuple:

        hullList = []
        # Convert to RGB to draw contour on - shouldn't recreate every time
        self.display = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR, dst=self.display)

        _, cnts, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(cnts) >= 1:
            acceptable_cnts = []
            # Check if the found contour is possibly a target
            for current_contour in enumerate(cnts):
                area = cv2.contourArea(current_contour[1])
                if PP_MAX_CONTOUR_AREA > area > PP_MIN_CONTOUR_AREA:
                    box = cv2.boundingRect(current_contour[1])
                    # Convex hull gives the bounding polygon of the contour with no
                    # interior angles greater than 180deg
                    hull = cv2.convexHull(current_contour[1])
                    hull_area = cv2.contourArea(hull)
                    # If the contour takes up more than X% of the Hull and
                    # width greater than height
                    if (
                        PP_MAX_AREA_RATIO > area / hull_area > PP_MIN_AREA_RATIO
                        and box[2] > box[3]
                    ):
                        # print(box) # X,Y,W,H
                        # print("P %.2f, Area %d, Hull, %d" % (area / hull_area, area, hull_area))
                        acceptable_cnts.append(current_contour[1])
                        hullList.append(hull)

            if acceptable_cnts:
                if len(acceptable_cnts) > 1:
                    # Pick the largest found 'power port'
                    power_port_contour = max(
                        acceptable_cnts, key=lambda x: cv2.contourArea(x)
                    )
                else:
                    power_port_contour = acceptable_cnts[0]
                power_port_points = get_corners_from_contour(power_port_contour)
                if len(power_port_points) != 4:
                    return None
                return power_port_points
            else:
                return None
        else:
            return None

    def create_annotated_display(self, frame: np.ndarray, points: np.ndarray):
        cv2.drawContours(
            frame, points.reshape(1, 4, 2), -1, self.COLOUR_BLUE, thickness=2
        )
        for point in points:
            cv2.circle(frame, tuple(point[0]), 5, self.COLOUR_BLUE, thickness=2)

        return frame

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
            midX = self.get_mid(power_port)

            target_top = min(list(power_port[:, :, 1]))
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

        frame_time, self.frame = self.camera_manager.get_frame(0)
        if frame_time == 0:
            error = self.camera_manager.get_error()
            self.camera_manager.source.notifyError(error)
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
    vision = Vision(CameraManager(), NTConnection())
    while True:
        vision.run()
