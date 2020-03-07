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
from typing import Optional, Tuple


class VisionTarget:
    def __init__(self, contour: np.ndarray) -> None:
        """Initialise a vision target object
        
        Args:
            contour: a single numpy/opencv contour
        """
        self.contour = contour.reshape(-1, 2)
        self._validate_and_reduce_contour()

    def _validate_and_reduce_contour(self):
        self.is_valid_target = True

    def get_leftmost_x(self) -> int:
        return min(list(self.contour[:, 0]))

    def get_rightmost_x(self) -> int:
        return max(list(self.contour[:, 0]))

    def get_middle_x(self) -> int:
        """ Use the cv2 moments to find the centre x of the contour.
        We just copied it from the opencv reference. The y is just the lowest
        pixel in the image."""
        M = cv2.moments(self.contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
        else:
            cX = 160
        return cX


    def get_highest_y(self) -> int:
        return min(list(self.contour[:, 1]))

    def get_lowest_y(self) -> int:
        return max(list(self.contour[:, 1]))


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
def _tilt_factor_to_radians(value, half_zoomed_fov_height) -> float:
    # The following number is the amount of fov height, in metres, we have
    # available to tilt through in one direction, so we scale the tilt value
    # from it's range (-10 - 10), to plus or minus this value.
    # Positive tilt moves the view down the image, which is negative vertical
    # angle, so the two ranges are inverted.
    vertical_fov_excursion = MAX_FOV_HEIGHT / 2 - half_zoomed_fov_height
    return scale_value(
        value, -10.0, 10.0, vertical_fov_excursion, -vertical_fov_excursion, 1.0,
    )


class Vision:
    """Main vision class.

    """

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

    MARGIN = 30
    # We want to zoom such that we scale the excursion to the following maximum,
    # Where excursion is the distance from the centre of the image to either
    # min_x or max_x of the target, whichever is fartest from the image centre.
    MAX_EXCURSION = FRAME_WIDTH / 2 - MARGIN

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

    def reset_target_extrema(self):
        # Previous target parameters are in pixels and are used to determine
        # new zoom and tilt settings for the next frame. Note that these are
        # valid only for the current zoom and tilt.
        self.previous_target_min_x = None
        self.previous_target_max_y = None
        self.previous_target_top = None

    def reset_zoom_and_tilt(self):
        self.reset_target_extrema()
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
        # New settings don't take until two frames later, so skip one now
        self.camera_manager.get_frame()

    def adjust_zoom_and_tilt(self):
        if (
            self.previous_target_min_x is not None
            and self.previous_target_max_x is not None
            and self.previous_target_top is not None
        ):
            # First we compute the new zoom
            # A length in pixels l1 at zoom z1 is related to the length in pixels
            # l2 at zoom z2 by
            # l1/z1 = l2/z2, or l1/l2 = z1/z2, or z2 = z1*l2/l1
            # To compute a new zoom, we want to scale a length l at zoom z1 to a
            # max length l2 at the new zoom z2. Using the above:
            # new_zoom = old_zoom * l2 / l1
            # The length we want to scale is the excursion from the centre of the
            # image, and the length we want to scale to is MAX_EXCURSION, clipped
            # to the maximum and a minimum of 1.
            min_from_centre = self.previous_target_min_x - FRAME_WIDTH / 2
            max_from_centre = self.previous_target_max_x - FRAME_WIDTH / 2
            excursion = max(abs(min_from_centre), abs(max_from_centre))
            new_zoom = round(self.zoom_factor * self.MAX_EXCURSION / excursion, 2)
            # round to 2 decimal places because we'll be multiplying by 100
            if new_zoom > self.MAX_ZOOM_FACTOR:
                new_zoom = self.MAX_ZOOM_FACTOR
            if new_zoom < self.MIN_ZOOM_FACTOR:
                new_zoom = self.MIN_ZOOM_FACTOR
            # Now we compute the new tilt. We want to put the top of the target
            # as close as possible to the centre of the image.
            # Don't bother with tilt if we aren't zoomed in
            if new_zoom - self.MIN_ZOOM_FACTOR > self.MIN_ZOOM_DELTA:
                # Compute y position of the top of the target in the current zoom and
                # tilt space
                # The pixels above the frame at the current zoom and 0 tilt:
                extra_at_top = FRAME_HEIGHT / 2 * (self.zoom_factor - 1.0)
                increment = extra_at_top / self.NUM_TILT_INCREMENTS
                total_current_y = (
                    self.previous_target_top + extra_at_top + self.tilt_factor * increment
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
                new_tilt = 0 # because zoom is so close to 1.0
            # Finally set and use the new values if either has changed
            print(
                "old zoom: ",
                self.zoom_factor,
                " new zoom: ",
                new_zoom,
                " old tilt: ",
                self.tilt_factor,
                " new tilt: ",
                new_tilt,
            )
            if (
                abs(new_zoom - self.zoom_factor) > self.MIN_ZOOM_DELTA
                or new_tilt != self.tilt_factor
            ):
                print("This is different, so setting camera")
                self.zoom_factor = new_zoom
                self.tilt_factor = new_tilt
                self.set_camera_zoom_and_tilt()

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

            self.previous_target_top = target_top
            self.previous_target_min_x = min(list(power_port[1][:, :, 0]))[0]
            self.previous_target_max_x = max(list(power_port[1][:, :, 0]))[0]
            print(f"target top: {target_top}, min_x: {self.previous_target_min_x}, max_x: {self.previous_target_max_x}")
            # target_bottom = max(list(power_port[:, :, 1]))
            # print("target top: ", target_top, " target bottom: ", target_bottom)
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
            print("angle: ", math.degrees(vert_angle), " distance: ", distance)

            return (distance, horiz_angle)
        else:
            print("no power port, so resetting extrema")
            self.reset_target_extrema()
            return None

    def run(self):

        """Main process function.
        Captures an image, processes the image, and sends results to the RIO.
        """
        self.connection.pong()

        self.adjust_zoom_and_tilt()  # based on previous frame values
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
