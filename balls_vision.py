import cv2
import numpy as np
from magic_numbers import *
from camera_manager import WebcamCameraManager, MockImageManager, CameraManager
from connection import NTConnection, DummyConnection
from utilities.functions import *
import math
import time
from power_port_vision import VisionTarget

from typing import Optional


class Ball(VisionTarget):
    def get_area(self) -> int:
        return cv2.contourArea(self.contour)


class Vision:
    BALL_HSV_LOWER_BOUND = (25, 50, 60)
    BALL_HSV_UPPER_BOUND = (40, 255, 240)

    MIN_CNT_SIZE = 50
    MAX_CNT_SIZE = 1000000

    def __init__(self, camera_manager: CameraManager, connection: NTConnection) -> None:
        # Memory allocation
        self.hsv = np.zeros(shape=(FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        self.display = self.hsv.copy()
        self.mask = np.zeros(shape=(FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)

        # Camera config
        self.camera_manager = camera_manager
        self.camera_manager.set_camera_property("white_balance_temperature_auto", 0)
        self.camera_manager.set_camera_property("exposure_auto", 1)
        self.camera_manager.set_camera_property("focus_auto", 0)
        self.camera_manager.set_camera_property("exposure_absolute", 1)
        self.camera_manager.set_camera_property("zoom_absolute", 100)

        self.connection = connection

        self.old_fps_time = 0

        self.last_path = None
        self.path_confidence = 0
        self.confidence_threshold = (
            -1
        )  # have to see the same path this many times to be sure (-1 for no delay)

    def read_data(self, file="balls_data.npz"):
        # reads the data from the file to use with the knn
        with open(file, "rb") as f:
            self._data = np.load(f)
            self.data = self._data["balls"].astype(np.float32)
            self.labels = self._data["labels"].astype(np.float32)
        self.knn = cv2.ml.KNearest_create()
        self.knn.train(self.data, cv2.ml.ROW_SAMPLE, self.labels)

    def create_annotated_display(self, frame, balls):
        # draws the balls and ball contours onto the frame
        cv2.drawContours(
            frame,
            [b.contour for b in balls],
            -1,
            (255, 0, 0),
            thickness=2,
        )

        for b in balls:
            cv2.circle(
                frame,
                (int(b.get_middle_x()), int(b.get_middle_y())),
                int(math.sqrt(b.get_area())),
                (0, 255, 0),
                2,
            )
        return frame

    def find_balls(self, frame: np.ndarray):
        # finds five largest yellowish contours
        # returns: returns the position of each ball as an offest from the average of their positions
        self.hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV, dst=self.hsv)
        self.mask = cv2.inRange(
            self.hsv,
            self.BALL_HSV_LOWER_BOUND,
            self.BALL_HSV_UPPER_BOUND,
            dst=self.mask,
        )
        kernel = np.ones((5, 5), np.uint8)
        self.mask = cv2.dilate(self.mask, kernel, iterations=1)
        self.mask = cv2.erode(self.mask, kernel, iterations=1)
        *_, cnts, _ = cv2.findContours(
            self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if len(cnts) >= 1:
            acceptable_targets = []
            # Check if the found contour is possibly a target
            for current_contour in cnts:
                if (
                    self.MIN_CNT_SIZE
                    < cv2.contourArea(current_contour)
                    < self.MAX_CNT_SIZE
                ):
                    acceptable_targets.append(Ball(current_contour))

            self.balls = sorted(acceptable_targets, key=lambda x: x.get_area())[
                0:3
            ]  # takes the 3 largest targets
            balls_output = []
            for target in self.balls:
                i = [target.get_middle_x(), target.get_middle_y(), target.get_area()]
                i = list(map(int, i))
                balls_output.append(i)

        self.display = frame.copy()
        self.display = self.create_annotated_display(self.display, self.balls)
        return balls_output

    def find_path(self, frame: np.ndarray):
        # from the ball positions it catagorizes it as one of the four paths
        # uses K Nearest Neighbour with labeled data stored in balls_data.npz
        # returns: the catagorizations as a int and the angle in radians
        balls = self.find_balls(frame)
        if len(balls) == 3:
            angle = get_horizontal_angle(
                sum(b[0] for b in balls), FRAME_WIDTH, MAX_FOV_WIDTH / 2
            )
            balls_offsets = self.normalize(balls)

            ret, result, neighbours, dist = self.knn.findNearest(np.array([balls_offsets]), k=3)
            if self.last_path == ret:
                self.path_confidence += 1
            else:
                self.path_confidence = 0
            self.last_path = ret

            if self.path_confidence > self.confidence_threshold:
                return ret, angle
        else:
            print("not enough balls")

    @staticmethod
    def normalize(balls: list):
        # normalizes the positions of balls to be offsets from 0 in pixels, retains size
        avg = (
            sum([x[0] for x in balls]) / len(balls),
            sum([x[1] for x in balls]) / len(balls),
        )
        shifted_balls = [(x[0] - avg[0], x[1] - avg[1], x[2]) for x in balls]
        return np.reshape(np.array(shifted_balls), len(balls) * 3).astype(np.float32)

    def run(self):
        self.connection.pong()
        frame_time, frame = self.camera_manager.get_frame()

        if frame_time == 0:
            self.camera_manager.notify_error(self.camera_manager.get_error())
            return
        # Flip the image cause originally upside down.
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        results = self.find_path(frame)
        self.connection.set_fps()
        if results is not None:
            path, angle = results
            print(f"sending {path} {angle}")
            self.connection.send_results(
                (path, angle, time.monotonic())
            )  # path, angle (radians), timestamp
        self.camera_manager.send_frame(self.display)


if __name__ == "__main__":
    test = True

    if test:
        im = cv2.imread("tests/balls/B2-0.jpg")
        vision = Vision(
            MockImageManager(im, display_output=True), NTConnection()
        )  # WebcamCameraManager(1)
        vision.read_data()
        for i in range(1):
            vision.run()
            time.sleep(0.1)

    else:
        vision = Vision(
            CameraManager("Power Port Camera", "/dev/video0", 240, 320, 30, "kYUYV"),
            NTConnection(),
        )
        while True:
            vision.run()
