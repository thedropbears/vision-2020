from connection import Connection
from camera_manager import CameraManager
import cv2
import numpy as np


class Vision:
    def __init__(self, connection: Connection, camera_manager: CameraManager) -> None:
        self.camera_manager = camera_manager
        self.connection = Connection

    def run(self) -> None:
        """Run the main vision loop once."""
        frame_time, self.frame = self.camera_manager.get_frame()
        if not frame_time:
            error = self.camera_manager.get_error()
            self.camera_manager.notify_error(error)
