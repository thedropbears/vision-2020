import numpy as np
import json
import cv2
from magic_numbers import *


class CameraManager:
    CONFIG_FILE_PATH = "/boot/frc.json"

    def __init__(self):
        """Initialises a Camera Manager"""
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

        Returns:
            A list of dictionaries containing the name, path, and config info
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
        """Initialises a camera.

        Args:
            config: A dictionary with keys "name", "path", and "config"
                as found by the read_config_json() function
        Returns:
            A cv2 VideoSource
        """
        camera = self.cs.startAutomaticCapture(name=config["name"], path=config["path"])
        camera.setConfigJson(json.dumps(config["config"]))
        return camera

    def get_frame(self, camera: int = 0):
        """Gets a frame from the specified camera.

        Args:
            camera: Which camera to get the frame from. Default is 0.
        Returns:
            Frame_time, or 0 on error.
            A numpy array of the frame, dtype=np.uint8, BGR.
        """
        frame_time, self.frame = self.sinks[camera].grabFrameNoTimeout(image=self.frame)
        return frame_time, self.frame

    def send_frame(self, frame: np.ndarray):
        """Sends a frame to the driver display.

        Args:
            frame: A numpy array image. (Should always be the same size)
        """
        self.source.putFrame(frame)


class DummyImageManager:
    def __init__(self, image: np.ndarray):
        """Initialises a Dummy Image Manager

        Args:
            image: A BGR numpy image array
        """
        self.image = image

    def change_image(self, new_image: np.ndarray):
        """Changes self.image.

        Args:
            new_image: The new image to switch to.
        """
        self.image = new_image

    def get_frame(self, camera: int = 0):
        """Returns self.image.

        Args:
            camera: Not needed, just here to ensure similarity with CameraManager.
        Returns:
            self.image, a BGR numpy array.
        """
        return self.image

    def send_frame(self, frame: np.ndarray):
        ...


class DummyVideoManager:
    def __init__(self, video: cv2.VideoCapture):
        """Initialises a Dummy Video Manager.

        Args:
            video: An opencv video, as received by cv2.VideoCapture
        """
        self.video = video

    def get_frame(self, camera: int = 0):
        """Returns the next frame of self.video.

        Args:
            camera: Not needed, just here to ensure similarity with CameraManager.
        Returns:
            The next frame of self.video.
        """
        return self.video.read()[1]

    def send_frame(self, frame: np.ndarray):
        ...


class DummyCameraManager:
    def __init__(self, camera: int = 0):
        """Initialises a Dummy Camera Manager. Designed to run on a non-pi computer.
        Initialises it with the first detected system camera, for example a webcam.
        
        Args:
            camera: Which camera to use. Default is 0th, probably a builtin webcam for most people.
        """
        self.video = cv2.VideoCapture(camera)

    def get_frame(self, camera: int = 0):
        """Returns the current video frame.

        Args:
            camera: Not needed, just here to ensure similarity with CameraManager.
        Returns:
            The current video frame.
        """

    def send_frame(self, frame: np.ndarray):
        ...
