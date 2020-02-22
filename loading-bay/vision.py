from connection import Connection
from camera_manager import CameraManager
import cv2
import numpy as np
import time


class Vision:
    def __init__(self, connection: Connection, camera_manager: CameraManager) -> None:
        self.camera_manager = camera_manager
        self.connection = connection

        self.image = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), np.uint8)

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

    def get_image_values(self, frame: np.ndarray):
        return None
