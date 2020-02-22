from connection import Connection
from camera_manager import CameraManager
import sys


class Vision:
    def __init__(self, connection: Connection, camera_manager: CameraManager):
        self.camera_manager = camera_manager

    def run(self):
        frame_time, self.frame = self.camera_manager.get_frame()
        if not frame_time:
            error = self.camera_manager.get_error()
            self.camera_manager.notify_error(error)

