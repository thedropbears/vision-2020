import argparse
import cv2
import magic_numbers
import numpy as np
from camera_manager import MockImageManager, MockVideoManager, WebcamCameraManager
from connection import DummyConnection
from power_port_vision import Vision

parser = argparse.ArgumentParser(
    "Simulate code by giving it an image, list of images, video, or camera."
)
parser.add_argument("-v", "--video", help="Use a video")
parser.add_argument(
    "-i",
    "--image",
    nargs="+",
    help="Specify an image or list of images. for example, `sim.py --image foo.jpg bar.jpg",
)
parser.add_argument(
    "-c",
    "--camera",
    help="Use a camera. Must be followed by a camera number. For example, most webcams would be 0.",
)
parser.add_argument(
    "-pp", "--power-port", help="Simulate with power port code", action="store_true"
)
args = parser.parse_args()

if args.image:
    print(f"IMAGE: {args.image}")
    frame = np.zeros(
        shape=(magic_numbers.FRAME_HEIGHT, magic_numbers.FRAME_WIDTH, 3), dtype=np.uint8
    )
    current_image = 0
    camera_manager = MockImageManager(frame, display_output=True)

elif args.video:
    print(f"VIDEO: {args.video}")
    camera_manager = MockVideoManager(cv2.VideoCapture(args.video), display_output=True)

elif args.camera:
    print(f"CAMERA: {args.camera}")
    camera_manager = WebcamCameraManager(int(args.camera))

connection = DummyConnection()
vision = Vision(camera_manager, connection)

while True:
    if args.image:
        frame = cv2.imread(args.image[current_image])
        current_image += 1
        camera_manager.change_image(frame)
        if current_image == len(args.image):
            current_image = 0
    vision.run()
