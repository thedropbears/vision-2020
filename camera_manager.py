"""The CameraManager class for The Drop Bears' vision code"""

import json
import numpy as np
from magic_numbers import *
import cv2


class CameraManager:
    CONFIG_FILE_PATH = "/boot/frc.json"

    def __init__(self, test_img=None, test_video=None, test_display=False):
        self.testing = False
        self.frame = None
        self.video = None
        self.test_display = test_display
        if not type(test_img) == type(None):
            self.testing = True
            self.frame = test_img
        elif not type(test_video) == type(None):
            self.testing = True
            self.video = test_video
        else:
            from cscore import CameraServer

            self.cs = CameraServer.getInstance()
            self.camera_configs = self.read_config_JSON()

            self.cameras = [
                self.start_camera(camera_config)
                for camera_config in self.camera_configs
            ]
            for prop in self.cameras[0].enumerateProperties():
                print(prop.getName(), prop.getKind())

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
            print(j)

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
        # print(camera.enumerateProperties())
        return camera

    def get_frame(self, camera: int) -> tuple:
        """Grabs a frame from the specified sink"""
        if self.testing:
            if self.video:
                self.frame = self.video.read()[1]
                return 1, self.frame
            else:
                return 1, self.frame
        frame_time, self.frame = self.sinks[camera].grabFrameNoTimeout(image=self.frame)
        #self.frame = cv2.flip(self.frame, 0)
        return frame_time, self.frame

    def send_frame(self, frame):
        if self.test_display:
            cv2.imshow("frame", self.frame)
            cv2.imshow("image", frame)
            cv2.waitKey(0)
        else:
            self.source.putFrame(frame)

    def setCameraProperty(self, camera, property, value):
        self.cameras[camera].getProperty(property).set(value)
