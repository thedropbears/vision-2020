import numpy as np
import json
import cv2
from magic_numbers import *
from typing import List, Tuple, Dict
import sys


class CameraManager:
    def __init__(self, camera_configs: List[Dict]) -> None:
        """Initialises a Camera Manager
        
        Args:
            camera_configs: A list of dictionaries with the cameras' info. For an example, see power_port_vision.py
        """
        from cscore import CameraServer

        self.cs = CameraServer.getInstance()
        self.camera_configs = camera_configs

        self.cameras = [
            self.start_camera(camera_config) for camera_config in self.camera_configs
        ]

        # In this, source and sink are inverted from the cscore documentation.
        # self.sink is a CvSource and self.sources are CvSinks. This is because it makes more sense for a reader.
        # We get images from a source, and put images to a sink.
        self.sources = [self.cs.getVideo(camera=camera) for camera in self.cameras]
        self.sink = self.cs.putVideo("Driver_Stream", FRAME_WIDTH, FRAME_HEIGHT)
        # Width and Height are reversed here because the order of putVideo's width and height
        # parameters are the opposite of numpy's (technically it is an array, not an actual image).
        self.frame = np.zeros(shape=(FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

    def start_camera(self, config: Dict):
        """Initialises a camera.

        Args:
            config: A dictionary with keys "name", "path", and "config"
        Returns:
            A cv2 Videosink
        """
        camera = self.cs.startAutomaticCapture(name=config["name"], path=config["path"])
        camera.setConfigJson(json.dumps(config["config"]))
        return camera

    def get_frame(self, camera: int = 0) -> Tuple[float, np.ndarray]:
        """Gets a frame from the specified camera.

        Args:
            camera: Which camera to get the frame from. Default is 0.
        Returns:
            Frame_time, or 0 on error.
            A numpy array of the frame, dtype=np.uint8, BGR.
        """
        frame_time, self.frame = self.sources[camera].grabFrameNoTimeout(
            image=self.frame
        )
        return frame_time, self.frame

    def send_frame(self, frame: np.ndarray) -> None:
        """Sends a frame to the driver display.

        Args:
            frame: A numpy array image. (Should always be the same size)
        """
        self.sink.putFrame(frame)

    def get_error(self, camera: int = 0) -> str:
        """Gets an error from the camera.
        Should be run by Vision when frame_time is 0.

        Args:
            camera: Which camera to get the error from.
        Returns:
            A string containing the camera's error.
        """
        return self.sources[camera].getError()

    def notify_error(self, error: str) -> None:
        """Sends an error to the console and the sink.
        Args:
            error: The string to send. Should be gotten by get_error().

        """
        print(error, file=sys.stderr)
        self.sink.notifyError(error)

    def set_camera_property(self, camera, property, value) -> None:
        self.cameras[camera].getProperty(property).set(value)


class MockImageManager:
    def __init__(self, image: np.ndarray) -> None:
        """Initialises a Mock Image Manager

        Args:
            image: A BGR numpy image array
        """
        self.image = image

    def change_image(self, new_image: np.ndarray) -> None:
        """Changes self.image.

        Args:
            new_image: The new image to switch to. Should be a numpy image array.
        """
        self.image = new_image

    def get_frame(self, camera: int = 0) -> Tuple[float, np.ndarray]:
        """Returns self.image.

        Args:
            camera: Not needed, just here to ensure similarity with CameraManager.
        Returns:
            0.1: Simulates the frame_time
            self.image, a BGR numpy array.
        """
        return 0.1, self.image.copy()

    def send_frame(self, frame: np.ndarray):
        cv2.imshow("Image", frame)
        cv2.waitKey(0)

    def get_error(self, camera: int = 0) -> str:
        return "Error"

    def notify_error(self, error: str) -> None:
        """Prints an error to the console.
        Args:
            error: The string to print.

        """
        print(error, file=sys.stderr)

    def set_camera_property(self, camera, property, value) -> None:
        ...


class MockVideoManager:
    def __init__(self, video: cv2.VideoCapture):
        """Initialises a Mock Video Manager.

        Args:
            video: An opencv video, as received by cv2.VideoCapture
        """
        self.video = video

    def get_frame(self, camera: int = 0) -> Tuple[float, np.ndarray]:
        """Returns the next frame of self.video.

        Args:
            camera: Not needed, just here to ensure similarity with CameraManager.
        Returns:
            Whether or not it was successful. False means error.
            The next frame of self.video.
        """
        result = self.video.read()
        if result[0]:
            return result
        else:  # If we reach the end of the video, go back to the beginning.
            self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return self.video.read()

    def send_frame(self, frame: np.ndarray) -> None:
        ...

    def get_error(self, camera: int = 0) -> str:
        return "Error"

    def notify_error(self, error: str) -> None:
        """Prints an error to the console.
        Args:
            error: The string to print.

        """
        print(error, file=sys.stderr)

    def set_camera_property(self, camera, property, value) -> None:
        ...


class WebcamCameraManager:
    def __init__(self, camera: int = 0) -> None:
        """Initialises a Webcam Camera Manager. Designed to run on a non-pi computer.
        Initialises it with the first detected system camera, for example a webcam.
        
        Args:
            camera: Which camera to use. Default is 0th, probably a builtin webcam for most people.
        """
        self.video = cv2.VideoCapture(camera)

    def get_frame(self, camera: int = 0) -> Tuple[float, np.ndarray]:
        """Returns the current video frame.

        Args:
            camera: Not needed, just here to ensure similarity with CameraManager.
        Returns:
            Whether or not it was successful. False means error.
            The current video frame.
        """
        return self.video.read()

    def send_frame(self, frame: np.ndarray) -> None:
        cv2.imshow("image", frame)
        cv2.waitKey(1)

    def get_error(self, camera: int = 0) -> str:
        return "Error"

    def notify_error(self, error: str) -> None:
        """Prints an error to the console.
        Args:
            error: The string to print.

        """
        print(error, file=sys.stderr)

    def set_camera_property(self, camera, property, value) -> None:
        ...
