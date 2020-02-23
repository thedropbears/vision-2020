from camera_manager import CameraManager
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
    def __init__(self, camera_manager: CameraManager) -> None:
        self.camera_manager = camera_manager

        self.image = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), np.uint8)
        self.mask = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), np.uint8)
        self.hsv = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), np.uint8)

    def run(self) -> None:
        """Run the main vision loop once."""
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
        self.image = self.frame
        return None
