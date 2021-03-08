# pylint: disable=C0330
# pylint: disable=E1101

"""The Drop Bears' 2020 vision code.

This code is run on the Raspberry Pi 4. It is uploaded via the deploy script, e.g:
    python3 deploy.py --power-port
It can be found at https://github.com/thedropbears/vision-2020
"""
import cv2
import numpy as np
from connection import NTConnection
from camera_manager import CameraManager
from magic_numbers import *
from utilities.functions import *
import math
import time
from vision_target import VisionTarget
from typing import Optional


class PowerPort(VisionTarget):
    def _validate_and_reduce_contour(self):
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

    def get_middle_x_new(self) -> int:
        self.corners = get_corners_from_contour(self.contour)  # gets the four corners
        self.corners = sorted(
            self.corners, key=lambda x: x[0][0]
        )  # sort them by x position
        left_height = np.linalg.norm(
            self.corners[0] - self.corners[1]
        )  # find difference between two left corners
        right_height = np.linalg.norm(
            self.corners[2] - self.corners[3]
        )  # difference of two right corners
        # print("heights ", right_height, left_height)
        # print("x's", self.get_rightmost_x(), self.get_leftmost_x())
        ratio = left_height / right_height
        # print(self.get_rightmost_x() * ratio + self.get_leftmost_x() * 1 - ratio)
        return self.get_rightmost_x() * ratio + self.get_leftmost_x() * 1 - ratio

    def get_middle_x(self) -> int:
        return (self.get_rightmost_x() + self.get_leftmost_x()) / 2


def _tilt_factor_to_radians(value, half_zoomed_fov_height) -> float:
    # The following number is the amount of fov height, in metres, we have
    # available to tilt through in one direction, so we scale the tilt value
    # from it's range (-10 - 10), to plus or minus this value.
    # Positive tilt moves the view down the image, which is negative vertical
    # angle, so the two ranges are inverted.
    vertical_fov_excursion = MAX_FOV_HEIGHT / 2 - half_zoomed_fov_height
    return scale_value(
        value,
        -10.0,
        10.0,
        vertical_fov_excursion,
        -vertical_fov_excursion,
        1.0,
    )


class Vision:
    """Main vision class."""

    entries = None
    COLOUR_GREEN = (0, 255, 0)
    COLOUR_BLUE = (255, 0, 0)
    COLOUR_RED = (0, 0, 255)
    COLOUR_YELLOW = (0, 255, 255)

    # Zoom factor is in the range 1.0 - 5.0, which are scaled to 100-500 when
    # sent to the camera
    MIN_ZOOM_FACTOR = 1.0
    MAX_ZOOM_FACTOR = 5.0
    # Tilt is in very weird values, ranging from -36000 to 36000, but in steps
    # of 3600, irrespective of zoom. So every zoom level other than 100 has
    # 10 steps of tilt above and 10 below 0.
    MAX_TILT_FACTOR = 10

    # Change zoom only if it differs by 5 or more in the zoom scale (100-500)
    MIN_ZOOM_DELTA = 0.05

    HORIZONTAL_MARGIN = 30
    VERTICAL_MARGIN = 30
    # We want to zoom such that we scale to one of the following
    # maxima, where excursion is the distance from the centre of the image to either
    # min or max x of the target, whichever is fartest from the image centre.
    MAX_HORIZONTAL_EXCURSION = FRAME_WIDTH / 2 - HORIZONTAL_MARGIN
    MAX_VERTICAL_SIZE = FRAME_HEIGHT - VERTICAL_MARGIN * 2

    NUM_TILT_INCREMENTS = 10  # In each direction, i.e. 10 , 10 down, from 0

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

        # The following must be scaled before sending to the camera
        self.zoom_factor = 1.0  # Scales the fov by the inverse of this factor
        self.tilt_factor = 0  # -10 up to 10 down, as an int
        self.reset_zoom_and_tilt()
        self.set_camera_zoom_and_tilt()

    def reset_zoom_and_tilt(self):
        self.previous_power_port = None
        if self.zoom_factor != 1.0 or self.tilt_factor != 0.0:
            self.zoom_factor = 1.0
            self.tilt_factor = 0.0
            self.set_camera_zoom_and_tilt()

    def set_camera_zoom_and_tilt(self):
        # Camera takes zoom values from 100 to 500
        self.camera_manager.set_camera_property(
            "zoom_absolute", int(self.zoom_factor * 100)
        )
        # Camera takes tilt values from -36000 to 36000
        self.camera_manager.set_camera_property(
            "tilt_absolute", int(self.tilt_factor * 3600)
        )
        # New settings don't take until two frames later, so skip two now
        self.camera_manager.get_frame()
        self.camera_manager.get_frame()

    def adjust_zoom_and_tilt(self):
        if self.previous_power_port is not None:
            # First we compute the new zoom
            # A length in pixels l1 at zoom z1 is related to the length in pixels
            # l2 at zoom z2 by
            # l1/z1 = l2/z2, or l1/l2 = z1/z2, or z2 = z1*l2/l1
            # To compute a new zoom, we want to scale a length l at zoom z1 to a
            # max length l2 at the new zoom z2. Using the above:
            # new_zoom = old_zoom * l2 / l1
            # The length we want to scale is either the horizontal excursion or
            # the height of the target, depending on whether the width or height
            # of the target is larger. The horizontal excurison is the distance
            # from the minimum or maximum x from the centre of the image, and
            # the length we want to scale to is the appropriate maximum, clipped
            # to the maximum and a minimum of 1.
            # TODO: This test doesn't seem to be exactly correct. Perhaps we
            # should be computing both zooms and choosing the smaller, as the
            # larger would cause the other dimension to exceed the margin.
            if (
                self.previous_power_port.get_width()
                > self.previous_power_port.get_height()
            ):
                left_from_centre = abs(
                    self.previous_power_port.get_leftmost_x() - FRAME_WIDTH / 2
                )
                right_from_centre = abs(
                    self.previous_power_port.get_rightmost_x() - FRAME_WIDTH / 2
                )
                horizontal_excursion = max(left_from_centre, right_from_centre)
                new_zoom = round(
                    self.zoom_factor
                    * self.MAX_HORIZONTAL_EXCURSION
                    / horizontal_excursion,
                    2,
                )
                # print("new zoom from horizontal: ", new_zoom)
            else:
                new_zoom = round(
                    self.zoom_factor
                    * self.MAX_VERTICAL_SIZE
                    / self.previous_power_port.get_height(),
                    2,
                )
                # print("new zoom from vertical: ", new_zoom)
            # round to 2 decimal places because we'll be multiplying by 100
            if new_zoom > self.MAX_ZOOM_FACTOR:
                new_zoom = self.MAX_ZOOM_FACTOR
            if new_zoom < self.MIN_ZOOM_FACTOR:
                new_zoom = self.MIN_ZOOM_FACTOR
            # Now we compute the new tilt. We want to put the centre of the target
            # as close as possible to the centre of the image.
            # Don't bother with tilt if we aren't zoomed in at least a minimum
            if new_zoom - self.MIN_ZOOM_FACTOR > self.MIN_ZOOM_DELTA:
                # Compute y position of the centre of the target in the current zoom and
                # tilt space
                # The pixels above the frame at the current zoom and 0 tilt:
                extra_at_top = FRAME_HEIGHT / 2 * (self.zoom_factor - 1.0)
                increment = extra_at_top / self.NUM_TILT_INCREMENTS
                total_current_y = (
                    self.previous_power_port.get_middle_y()
                    + extra_at_top
                    + self.tilt_factor * increment
                )
                # tilt and y are both positive down
                # Next we compute the corresponding total y in the new zoom space
                new_total_y = total_current_y * new_zoom / self.zoom_factor
                # And the same parameters in the new space
                new_extra_at_top = FRAME_HEIGHT / 2 * (new_zoom - 1.0)
                new_increment = new_extra_at_top / self.NUM_TILT_INCREMENTS
                # The new tilt we want is the difference between the centre of the
                # expanded frame and the new total y, in increments.
                new_expanded_centre_y = FRAME_HEIGHT * new_zoom / 2
                # cast to int to round toward zero
                new_tilt = int((new_total_y - new_expanded_centre_y) / new_increment)
                # And we clip
                if new_tilt > self.MAX_TILT_FACTOR:
                    new_tilt = self.MAX_TILT_FACTOR
                if new_tilt < -self.MAX_TILT_FACTOR:
                    new_tilt = -self.MAX_TILT_FACTOR
            else:
                new_tilt = 0  # because zoom is so close to 1.0
            # Finally set and use the new values if either has changed
            # print(
            #   "old zoom: ",
            #    self.zoom_factor,
            #    " new zoom: ",
            #    new_zoom,
            #    " old tilt: ",
            #    self.tilt_factor,
            #    " new tilt: ",
            #    new_tilt,
            # )
            if (
                abs(new_zoom - self.zoom_factor) > self.MIN_ZOOM_DELTA
                or new_tilt != self.tilt_factor
            ):
                # print("This is different, so setting camera")
                self.zoom_factor = new_zoom
                self.tilt_factor = new_tilt
                self.set_camera_zoom_and_tilt()

    def find_power_port(self, frame: np.ndarray) -> Optional[PowerPort]:
        # Convert to RGB to draw contour on - shouldn't recreate every time
        self.display = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR, dst=self.display)

        *_, cnts, _ = cv2.findContours(
            frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
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

        # cv2.circle(
        #     frame,
        #     (int(power_port.get_middle_x()), int(power_port.get_middle_y())),
        #     5,
        #     self.COLOUR_GREEN,
        #     thickness=2,
        # )  # new
        # cv2.circle(
        #     frame,
        #     (int(power_port.get_middle_x_old()), int(power_port.get_middle_y())),
        #     5,
        #     self.COLOUR_RED,
        #     thickness=2,
        # )  # old

        return frame

    def get_image_values(self, frame: np.ndarray) -> tuple:
        """Takes a frame, returns a tuple of results, or None."""
        self.hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV, dst=self.hsv)
        self.mask = cv2.inRange(
            self.hsv, HSV_LOWER_BOUND, HSV_UPPER_BOUND, dst=self.mask
        )

        power_port = self.find_power_port(self.mask)

        self.previous_power_port = power_port
        if power_port is not None:
            self.display = self.create_annotated_display(self.display, power_port)
            midX = power_port.get_middle_x()

            target_top = power_port.get_highest_y()

            self.previous_power_port = power_port
            zoomed_fov_height = MAX_FOV_HEIGHT / self.zoom_factor
            zoomed_fov_width = MAX_FOV_WIDTH / self.zoom_factor
            horiz_angle = get_horizontal_angle(
                midX, FRAME_WIDTH, zoomed_fov_width / 2, True
            )

            vert_angle = get_vertical_angle_linear(
                target_top, FRAME_HEIGHT, zoomed_fov_height / 2, True
            ) + _tilt_factor_to_radians(self.tilt_factor, zoomed_fov_height / 2)

            distance = get_distance(
                vert_angle, TARGET_HEIGHT_TOP, CAMERA_HEIGHT, GROUND_ANGLE
            )
            print(
                "horizontal angle: ", math.degrees(horiz_angle), " distance: ", distance
            )

            return (distance, horiz_angle)
        else:
            print("no power port")
            return None

    def run(self):

        """Main process function.
        Captures an image, processes the image, and sends results to the RIO.
        """
        self.connection.pong()

        self.adjust_zoom_and_tilt()  # based on previous power port
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
        else:
            self.reset_zoom_and_tilt()
        self.camera_manager.send_frame(self.display)


if __name__ == "__main__":
    vision = Vision(
        CameraManager("Power Port Camera", "/dev/video0", 240, 320, 30, "kYUYV"),
        NTConnection(),
    )
    while True:
        vision.run()
