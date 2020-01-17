"""The CameraManager class for The Drop Bears' vision code"""

import json
import numpy as np
from magic_numbers import *


class CameraManager:
    CONFIG_FILE_PATH = "/boot/frc.json"

    def __init__(self):
        from cscore import CameraServer

        self.cs = CameraServer.getInstance()
        self.camera_configs = self.read_config_JSON()

        self.cameras = [
            self.start_camera(camera_config) for camera_config in self.camera_configs
        ]

        self.sinks = [self.cs.getVideo(camera=camera) for camera in self.cameras]
        self.source = self.cs.putVideo("Driver_Stream", FRAME_WIDTH, FRAME_HEIGHT)

        self.frame = np.zeros(shape=(FRAME_WIDTH, FRAME_HEIGHT, 3), dtype=np.uint8)

    def read_config_JSON(self) -> list:
        """Reads camera config JSON.
        Returns a list of dictionaries containing the name, path, and config info
        of each camera in the config file.
        """
        with open(self.CONFIG_FILE_PATH) as json_file:
            j = json.load(json_file)

        cameras = j["cameras"]
        cameras = [
            {"name": camera["name"], "path": camera["path"], "config": camera}
            for camera in cameras
        ]

        return cameras

    def start_camera(self, config: dict):
        """Takes a VideoSource, returns a CvSink"""
        camera = self.cs.startAutomaticCapture(name=config["name"], path=config["path"])
        camera.setConfigJson(json.dumps(config["config"]))
        return camera

    def get_frame(self, camera: int) -> tuple:
        """Grabs a frame from the specified sink"""
        frame_time, self.frame = self.sinks[camera].grabFrameNoTimeout(image=self.frame)
        return frame_time, self.frame
