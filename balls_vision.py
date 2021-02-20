import cv2
import numpy as np
from magic_numbers import *
from camera_manager import WebcamCameraManager, MockImageManager
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

    def __init__(self, camera_manager) -> None:
        self.hsv = np.zeros(shape=(FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        self.display = self.hsv.copy()
        self.mask = np.zeros(shape=(FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)
        self.camera_manager = camera_manager

    def readData(self, file="balls_data.npz"):
        with open(file, "rb") as f:
            self._data = np.load(f)
            self.data = np.array([self._data["balls"]], dtype=np.int64)
            self.data.reshape((1, self.data.size))
            self.labels = np.array([self._data["labels"]])
        print(self.data, type(self.data), self.data.shape, self.data.dtype)
        print(self.labels, type(self.labels), self.labels.shape, self.labels.dtype)
        self.knn = cv2.ml.KNearest_create()
        self.knn.train(self.data, cv2.ml.ROW_SAMPLE, self.labels)
        ret, result, neighbours, dist = knn.findNearest(test, k=5)
        pass

    def find_balls(self):
        _, frame = self.camera_manager.get_frame()
        self.hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV, dst=self.hsv)
        self.mask = cv2.inRange(
            self.hsv,
            self.BALL_HSV_LOWER_BOUND,
            self.BALL_HSV_UPPER_BOUND,
            dst=self.mask,
        )
        kernel = np.ones((5, 5), np.uint8)
        cv2.imshow("old mask", self.mask)
        self.mask = cv2.dilate(self.mask, kernel, iterations=1)
        self.mask = cv2.erode(self.mask, kernel, iterations=1)
        cv2.imshow("new mask", self.mask)
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

            balls = sorted(acceptable_targets, key=lambda x: x.get_area())[
                0:3
            ]  # takes the 3 largest targets
            balls_output = []
            for target in balls:
                i = [target.get_middle_x(), target.get_middle_y(), target.get_area()]
                i = list(map(int, i))
                cv2.circle(frame, (i[0], i[1]), int(math.sqrt(i[2])), (0, 255, 0), 2)
                balls_output.append(i)
        cv2.imshow("frame", frame)
        # cv2.waitKey()
        return self.normalize(balls_output)

    @staticmethod
    def normalize(balls: list):
        # centers a list of ball pos's around 0
        avg = (
            sum([x[0] for x in balls]) / len(balls),
            sum([x[1] for x in balls]) / len(balls),
        )
        shifted_balls = [(x[0] - avg[0], x[1] - avg[1], x[2]) for x in balls]
        return shifted_balls

    def run():
        balls = self.find_balls()
        if len(balls) == 3:
            cv2.knn()


if __name__ == "__main__":
    im = cv2.imread("tests/balls/A1-0.jpg")
    # cv2.imshow("t", im)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    vision = Vision(MockImageManager(im))  # WebcamCameraManager(1)
    vision.readData()
    while True:
        vision.run()
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        # vision.BALL_HSV_UPPER_BOUND = tuple(map(int, input("enter upper ").split(" ")))
        # vision.BALL_HSV_LOWER_BOUND = tuple(map(int, input("enter lower ").split(" ")))
        # print(vision.BALL_HSV_LOWER_BOUND, vision.BALL_HSV_UPPER_BOUND)
    vision.camera_manager.video.release()
