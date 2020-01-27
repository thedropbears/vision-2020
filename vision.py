# pylint: disable=C0330
# pylint: disable=E1101

"""The Drop Bears' 2020 vision code.

This code is run on the Raspberry Pi 4. It is uploaded via the browser interface.
It can be found at https://github.com/thedropbears/vision-2020
"""
import sys
import cv2
import numpy as np
from connection import Connection
from camera_manager import CameraManager
from magic_numbers import *
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

    def __init__(
        self,
        test_img=None,
        test_video=None,
        test_display=False,
        using_nt=False,
        zooming=False,
    ):
        # self.entries = entries
        # Memory Allocation
        self.hsv = np.zeros(shape=(FRAME_WIDTH, FRAME_HEIGHT, 3), dtype=np.uint8)
        self.image = self.hsv.copy()
        self.mask = np.zeros(shape=(FRAME_WIDTH, FRAME_HEIGHT), dtype=np.uint8)

        # Camera Configuration
        self.CameraManager = CameraManager(
            test_img=test_img, test_video=test_video, test_display=test_display
        )

        self.Connection = Connection(using_nt=using_nt, test=test_video or test_img)
        self.zoom = 100

        self.testing = not (
            type(test_img) == type(None) or type(test_video) == type(None)
        )
        self.zoom = 100
        self.lastZoom = 100
        self.zooming = zooming

    def find_polygon(self, contour: np.ndarray, n_points: int = 4):
        """Finds the polygon which most accurately matches the contour.

        Args:
            contour (np.ndarray): Should be a numpy array of the contour with shape (1, n, 2).
            n_points (int): Designates the number of corners which the polygon should have.

        Returns:
            np.ndarray: A list of points representing the polygon's corners.
        """
        coefficient = CONTOUR_COEFFICIENT
        for _ in range(20):
            epsilon = coefficient * cv2.arcLength(contour, True)
            poly_approx = cv2.approxPolyDP(contour, epsilon, True)
            hull = cv2.convexHull(poly_approx)
            if len(hull) == n_points:
                return hull
            if len(hull) > n_points:
                coefficient += 0.01
            else:
                coefficient -= 0.01
        return None

    def find_loading_bay(self, frame: np.ndarray):
        cnts, hierarchy = cv2.findContours(
            self.mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )
        cnts = np.array(cnts)
        hierarchy = np.array(hierarchy)[0]
        outer_rects = {}
        inner_rects = {}

        for i, cnt in enumerate(cnts):
            if hierarchy[i][3] == -1:
                outer_rects[i] = (
                    self.find_polygon(cnt),
                    hierarchy[i],
                    cv2.contourArea(cnt),
                )
            else:
                inner_rects[i] = (
                    self.find_polygon(cnt),
                    hierarchy[i],
                    cv2.contourArea(cnt),
                )
        if not (inner_rects and outer_rects):
            return None

        good = []

        for i in outer_rects:
            if outer_rects[i][2] > MIN_CONTOUR_AREA:
                current_inners = []
                next_child = outer_rects[i][1][2]
                while next_child != -1:
                    current_inners.append(inner_rects[next_child])
                    next_child = inner_rects[next_child][1][0]
                largest = max(current_inners, key=lambda x: x[2])
                if (
                    abs((outer_rects[i][2] / largest[2]) - INNER_OUTER_RATIO) < 0.5
                    and abs(
                        (cv2.contourArea(outer_rects[i][0]) / outer_rects[i][2]) - 1
                    )
                    < RECT_AREA_RATIO
                    and abs((cv2.contourArea(largest[0]) / largest[2]) - 1)
                    < RECT_AREA_RATIO
                ):
                    good.append((outer_rects[i], largest))

        self.image = frame.copy()
        for pair in good:
            self.image = cv2.drawContours(
                self.image, pair[0][0].reshape((1, 4, 2)), -1, (255, 0, 0), thickness=2
            )
            self.image = cv2.drawContours(
                self.image,
                pair[1][0].reshape((1, 4, 2)),
                -1,
                (255, 0, 255),
                thickness=1,
            )
        return (0.0, 0.0)

    def find_power_port(self, frame: np.ndarray):
        _, cnts, _ = cv2.findContours(frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(cnts) >= 1:
            largestCnt = max(cnts, key=lambda x: cv2.contourArea(x))
            largestSize = cv2.contourArea(largestCnt)
            x, y, w, h = cv2.boundingRect(largestCnt)
            return (x, y, w, h)
        else:
            return None

    def createAnnotatedDisplay(self, frame, results, printing=False):
        frame = cv2.rectangle(
            frame,
            (results[0], results[1]),
            (results[0] + results[2], results[1] + results[3]),
            (255, 0, 0),
        )

        if printing:
            print("distance ", end="")
            print(distance)
            print("angle ", end="")
            print(angle)
            print("width ", end="")
            print(results[2])

    def get_angle(self, X):
        return (
            ((X / FRAME_WIDTH) - 0.5) * MAX_FOV_WIDTH * self.zoom / 100
        )  # 33.18 degrees #gets the angle

    def get_distance(self, width, angle):
        distance = (
            PORT_DIMENTIONS[0]
            / math.tan((width / FRAME_WIDTH) * (MAX_FOV_WIDTH / 2))
            * (self.zoom / 100)
        )  # the current method this uses is not mathmetically correct, the correct method would use the law of cosines
        # this just uses a tan and then tries to correct itself
        distance -= angle * 1.9
        distance *= 0.6
        return distance

    def get_image_values(self, frame: np.ndarray) -> tuple:
        """Takes a frame, returns a tuple of results, or None."""
        self.hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV, dst=self.hsv)
        self.mask = cv2.inRange(
            self.hsv, HSV_LOWER_BOUND, HSV_UPPER_BOUND, dst=self.mask
        )

        results = self.find_power_port(self.mask)
        if results != None:
            midX = results[0] + results[2] / 2  # finds middle of target
            angle = self.get_angle(midX)
            distance = self.get_distance(results[2], angle)
            self.image = self.mask

            return (results, angle, distance)
        else:
            return None

    def run(self):
        """Main process function.
        When ran, takes image, processes image, and sends results to RIO.
        """
        frame_time, self.frame = self.CameraManager.get_frame(0)
        self.frame = cv2.flip(self.frame, 0)
        if frame_time == 0:
            print(self.CameraManager.sinks[0].getError(), file=sys.stderr)
            self.CameraManager.source.notifyError(
                self.CameraManager.sinks[0].getError()
            )
        else:
            results = self.get_image_values(self.frame)
            if results != None:
                self.createAnnotatedDisplay(self.image, results[0])
                self.Connection.send_results(
                    (results[2], results[1], time.monotonic())
                )  # distance (meters), angle (radians), timestamp

                if self.zooming == True:
                    self.lastZoom = self.zoom
                    self.zoom = self.translate(abs(results[1]), 0.45, 0, 100, 200)
                    if abs(self.lastZoom - self.zoom) > 20:
                        self.CameraManager.setCameraProperty(
                            0, "zoom_absolute", round(self.zoom)
                        )
                # print(results[2])
            self.CameraManager.send_frame(self.image)

    def translate(
        self, value, leftMin, leftMax, rightMin, rightMax
    ):  # https://stackoverflow.com/questions/1969240/mapping-a-range-of-values-to-another
        # Figure out how 'wide' each range is
        leftSpan = leftMax - leftMin
        rightSpan = rightMax - rightMin

        # Convert the left range into a 0-1 range (float)
        valueScaled = float(value - leftMin) / float(leftSpan)

        # Convert the 0-1 range into a value in the right range.
        return rightMin + (valueScaled * rightSpan)


if __name__ == "__main__":
    # testImg = cv2.imread("tests/power_port/9m.PNG")
    # These imports are here so that one does not have to install cscore
    # (a somewhat difficult project on Windows) to run tests.

    camera_server = Vision(using_nt=True, zooming=False)
    while True:
        camera_server.run()
