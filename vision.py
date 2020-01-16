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


class Vision:
    """Main vision class.

    An instance should be created, with test=False (default). As long as the cameras are configured
    correctly via the GUI interface, everything will work without modification required.
    This will not work on most machines, so tests of the main process function are
    the only tests that can be done without a Pi running the FRC vision image.
    """

    # Magic Numbers:

    # Order of Power Port points
    # (0)__    (0, 0)    __(3)
    #    \ \            / /
    #     \ \          / /
    #      \ \________/ /
    #      (1)________(2)

    PORT_POINTS = [  # Given in inches as per the manual
        [19.625, 0, 0],
        [19.625 / 2, -17, 0],
        [-19.625 / 2, -17, 0],
        [-19.625, 0, 0],
    ]
    PORT_POINTS = np.array(  # Converted to mm
        [(2.54 * i[0], 2.54 * i[1], 0) for i in PORT_POINTS], np.float32
    ).reshape((4, 1, 3))

    FRAME_HEIGHT = 240
    FRAME_WIDTH = 320
    HSV_LOWER_BOUND = (30, 120, 80)
    HSV_UPPER_BOUND = (100, 255, 240)
    MIN_CONTOUR_AREA = 500
    CONTOUR_COEFFICIENT = 0.05
    INNER_OUTER_RATIO = 3.62
    RECT_AREA_RATIO = 0.2

    FOCAL_LENGTH = 3.67  # mm
    SENSOR_WIDTH = 4.8  # mm
    SENSOR_HEIGHT = 3.6  # mm
    FX = FOCAL_LENGTH * FRAME_WIDTH / SENSOR_WIDTH
    FY = FOCAL_LENGTH * FRAME_HEIGHT / SENSOR_HEIGHT
    CX = FRAME_WIDTH / 2
    CY = FRAME_HEIGHT / 2
    INTR_MATRIX = np.array(
        [[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]], dtype=np.float32
    )
    DIST_COEFF = np.array([0, 0, 0, 0], dtype=np.float32)

    entries = None

    def __init__(self, test=False, using_nt=False):
        # Memory Allocation
        self.hsv = np.zeros(
            shape=(self.FRAME_WIDTH, self.FRAME_HEIGHT, 3), dtype=np.uint8
        )
        self.image = self.hsv.copy()
        self.mask = np.zeros(
            shape=(self.FRAME_WIDTH, self.FRAME_HEIGHT), dtype=np.uint8
        )

        if not test:
            self.Connection = Connection(using_nt=using_nt, entries=self.entries)

            # Camera Configuration
            self.CameraManager = CameraManager()

    def find_polygon(self, contour: np.ndarray, n_points: int = 4):
        """Finds the polygon which most accurately matches the contour.

        Args:
            contour (np.ndarray): Should be a numpy array of the contour with shape (1, n, 2).
            n_points (int): Designates the number of corners which the polygon should have.

        Returns:
            np.ndarray: A list of points representing the polygon's corners.
        """
        coefficient = self.CONTOUR_COEFFICIENT
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
            if outer_rects[i][2] > self.MIN_CONTOUR_AREA:
                current_inners = []
                next_child = outer_rects[i][1][2]
                while next_child != -1:
                    current_inners.append(inner_rects[next_child])
                    next_child = inner_rects[next_child][1][0]
                largest = max(current_inners, key=lambda x: x[2])
                if (
                    abs((outer_rects[i][2] / largest[2]) - self.INNER_OUTER_RATIO) < 0.5
                    and abs(
                        (cv2.contourArea(outer_rects[i][0]) / outer_rects[i][2]) - 1
                    )
                    < self.RECT_AREA_RATIO
                    and abs((cv2.contourArea(largest[0]) / largest[2]) - 1)
                    < self.RECT_AREA_RATIO
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
        return (0.0, 0.0)

    def get_image_values(self, frame: np.ndarray) -> tuple:
        """Takes a frame, returns a tuple of results, or None."""
        self.hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV, dst=self.hsv)
        self.mask = cv2.inRange(
            self.hsv, self.HSV_LOWER_BOUND, self.HSV_UPPER_BOUND, dst=self.mask
        )
        results = self.find_loading_bay(frame)
        return results

    def run(self):
        """Main process function.
        When ran, takes image, processes image, and sends results to RIO.
        """
        frame_time, self.frame = self.CameraManager.get_frame(0)
        if frame_time == 0:
            print(self.CameraManager.sinks[0].getError(), file=sys.stderr)
            self.CameraManager.source.notifyError(
                self.CameraManager.sinks[0].getError()
            )
        else:
            results = self.get_image_values(self.frame)
            self.CameraManager.source.putFrame(self.image)
            self.Connection.send_results(results)


if __name__ == "__main__":

    # These imports are here so that one does not have to install cscore
    # (a somewhat difficult project on Windows) to run tests.

    camera_server = Vision()
    while True:
        camera_server.run()
