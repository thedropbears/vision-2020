import cv2
import numpy as np
from magic_numbers import *
from camera_manager import WebcamCameraManager, MockImageManager, CameraManager
from connection import NTConnection, DummyConnection
from utilities.functions import *
import math
import time
from vision_target import VisionTarget
from typing import Optional


class Ball(VisionTarget):
    def get_area(self) -> int:
        return cv2.contourArea(self.contour)

    def get_width_height_ratio(self) -> int:
        return max(self.get_width()/self.get_height(), self.get_height()/self.get_width())

    def get_conf(self) -> float: # gets confidence that this is a ball with heuristics
        # is the area of the countour divided by its width height ratio all multiplied by
        # the distance from the top of the frame (to prevent detecting the intake roller)
        area_heur = self.get_area()
        WH_heur = 3*self.get_width_height_ratio()**3
        if self.get_lowest_y()/FRAME_HEIGHT < 0.1:
            y_heur = 0
        else:
            y_heur = ((self.get_lowest_y()/FRAME_HEIGHT)-0.1)**0.1
        self.conf = (area_heur / WH_heur) * y_heur
        return self.conf

class Vision:
    BALL_HSV_LOWER_BOUND = (15, 100, 100)
    BALL_HSV_UPPER_BOUND = (30, 255, 240)

    MIN_CNT_SIZE = 70
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
        self.camera_manager.set_camera_property("exposure_absolute", 5)
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
        # you have to run collector.py if you want to update the file with new data
        with open(file, "rb") as f:
            self._data = np.load(f)
            self.data = self._data["balls"].astype(np.float32)
            self.labels = self._data["labels"].astype(np.float32)
        self.knn = cv2.ml.KNearest_create()
        self.knn.train(self.data, cv2.ml.ROW_SAMPLE, self.labels)

    def create_annotated_display(self, frame, balls = [], path = 0, text_label = True):
        # draws the balls and ball contours onto the frame
        # also also draws a text label of the path catagorization if text_label is true
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

        if text_label:
            if path == 1 or path == 3:
                textCol = (200, 10, 10) # red
            elif path == 2 or path == 4:
                textCol = (10, 10, 200) # blue
            else:
                textCol = (200, 200, 200) # grey (didnt find one)
            cv2.putText(frame, getPathStr(path), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, textCol, 2)
        return frame

    def find_balls(self, frame: np.ndarray):
        # Thresholds the image and finds contours from the mask
        # Filters out contours not in MIN_CNT_SIZE and MAX_CNT_SIZE
        # Then turns the contours into Balls objects and takes the x,y positions of the three with the highest confidence's
        # returns: the three balls as x,y positions
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
        balls_output = []
        if len(cnts) >= 1:
            acceptable_targets = []
            # Filters out tiny and massive contours
            for current_contour in cnts:
                if (
                    self.MIN_CNT_SIZE
                    < cv2.contourArea(current_contour)
                    < self.MAX_CNT_SIZE
                ):
                    acceptable_targets.append(Ball(current_contour))

            self.balls = sorted(acceptable_targets, key=lambda x: x.get_conf())[-3:] # takes the 3 most confident

            self.balls = sorted(self.balls, key=lambda x:x.get_middle_y()) # sort them by y so they are always in the same order
            for target in self.balls: # gets their middles, int's them and adds them to balls_output
                i = [target.get_middle_x(), target.get_middle_y()]
                i = list(map(int, i))
                balls_output.append(i)

        return balls_output

    def find_path(self, frame: np.ndarray):
        # from the ball positions it catagorizes it as one of the four paths
        # uses K Nearest Neighbour with labeled data stored in balls_data.npz
        # returns: the catagorizations as a int (use getPathNum and getPathStr from magic nums to go between int and path name) 
        # and the angle in radians
        balls = self.find_balls(frame)
        if len(balls) == 3:
            angle = get_horizontal_angle(
                sum(b[0] for b in balls), FRAME_WIDTH, MAX_FOV_WIDTH / 2
            )
            balls_offsets = self.normalize(balls)
            ret, result, neighbours, dist = self.knn.findNearest(np.array([balls_offsets]), k=10)
            if self.last_path == ret:
                self.path_confidence += 1
            else:
                self.path_confidence = 0
            self.last_path = ret

            self.display = self.create_annotated_display(frame, self.balls, ret)
            # cv2.imshow("mask", self.mask)
            if self.path_confidence > self.confidence_threshold:
                return ret, angle
        else:
            self.display = self.create_annotated_display(self.display, self.balls)
            print("not enough balls")

    @staticmethod
    def normalize(balls: list):
        # normalizes the positions of balls to be offsets from 0 in pixels
        avg = (
            sum([x[0] for x in balls]) / len(balls),
            sum([x[1] for x in balls]) / len(balls),
        )
        shifted_balls = [(x[0] - avg[0], x[1] - avg[1]) for x in balls]
        return np.reshape(np.array(shifted_balls), len(balls) * 2).astype(np.float32)

    def run(self):
        self.connection.pong()
        frame_time, frame = self.camera_manager.get_frame()

        if frame_time == 0:
            self.camera_manager.notify_error(self.camera_manager.get_error())
            return
        # Flip the image cause originally upside down.
        # frame = cv2.rotate(frame, cv2.ROTATE_180)
        results = self.find_path(frame)
        print(results)
        self.connection.set_fps()
        if results is not None:
            path, angle = results
            self.connection.send_results(
                (path, angle, time.monotonic())
            )  # path, angle (radians), timestamp
        self.camera_manager.send_frame(self.display)


if __name__ == "__main__":
    test = False

    if test:
        import os

        imgs = os.listdir("tests/balls/test/")
        for i in imgs:
            print("actual", i)
            im = cv2.imread(os.path.join("tests/balls/test", i))
            vision = Vision(
                MockImageManager(im, display_output=True), NTConnection()
            )  # WebcamCameraManager(1)
            vision.read_data()
            for i in range(1):
                vision.run()
                time.sleep(0.1)
                plt.show()

            print("")

    else:
        vision = Vision(
            CameraManager("Power Port Camera", "/dev/video0", 240, 320, 30, "kYUYV"),
            NTConnection(),
        )
        vision.read_data()
        while True:
            vision.run()
