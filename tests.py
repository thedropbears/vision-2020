import cv2
import magic_numbers
import numpy as np
import power_port_vision
import pytest
import unittest
from camera_manager import MockImageManager
from connection import DummyConnection
from typing import Tuple
from utilities.functions import *

Results = Tuple[float, float, float]

# This file should be run from the command line with pytest.
# For example, on Windows, you might do `py -3 -m pytest tests.py`

class VisionTests(unittest.TestCase):

    files = (
        "./tests/power_port/4m1.png",
        "./tests/power_port/6m2.png",
        "./tests/power_port/7m2.png",
        "./tests/power_port/9m1.png",
    )
    expected_results = ((4, 0), (6, 0), (7, 0), (9, 0))

    def _test_power_port_image(self, filename: str, expected_results: Results):
        # Filename is relative
        self.frame = cv2.imread(filename)
        self.camera_manager.change_image(self.frame)
        self.vision.run()

        results = self.connection.results
        if results is not None:
            print(expected_results)
            self.assertAlmostEqual(results[0], float(expected_results[0]))
            self.assertAlmostEqual(results[1], float(expected_results[1]))

    @pytest.mark.xfail
    def test_power_port(self):
        self.frame = np.zeros(
            shape=(magic_numbers.FRAME_HEIGHT, magic_numbers.FRAME_WIDTH, 3),
            dtype=np.uint8,
        )
        self.camera_manager = MockImageManager(self.frame, display_output=False)
        self.connection = DummyConnection()
        self.vision = power_port_vision.Vision(self.camera_manager, self.connection)
        for filename, expected_results in zip(self.files, self.expected_results):
            self._test_power_port_image(filename, expected_results)


class UtilitiesTests(unittest.TestCase):
    TEST_INPUTS = np.array(
        [
            [
                [[67, 40]],
                [[161, 41]],
                [[258, 43]],
                [[238, 101]],
                [[211, 160]],
                [[179, 158]],
                [[146, 151]],
                [[122, 151]],
                [[86, 146]],
            ],
            [
                [[66, 65]],
                [[92, 57]],
                [[116, 50]],
                [[134, 63]],
                [[150, 79]],
                [[151, 100]],
                [[152, 121]],
                [[100, 132]],
                [[64, 116]],
            ],
        ]
    )
    TEST_OUTPUTS = np.array(
        [
            [
                np.array([[67, 40]], dtype=np.int32),
                np.array([[258, 43]], dtype=np.int32),
                np.array([[211, 160]], dtype=np.int32),
                np.array([[86, 146]], dtype=np.int32),
            ],
            [
                np.array([[66, 65]], dtype=np.int32),
                np.array([[116, 50]], dtype=np.int32),
                np.array([[150, 79]], dtype=np.int32),
                np.array([[152, 121]], dtype=np.int32),
                np.array([[100, 132]], dtype=np.int32),
                np.array([[64, 116]], dtype=np.int32),
            ],
        ]
    )
    INTR_MATRIX = np.array(
        [[320, 0.0, 160], [0.0, 320, 120], [0.0, 0.0, 1.0]], dtype=np.float32
    )

    DIST_COEFFS = np.array(
        [
            [
                1.27391079e-01,
                -5.09404111e-01,
                -7.87105714e-04,
                2.60450896e-03,
                1.04097100e00,
            ]
        ],
        dtype=np.float32,
    )

    def test_scale_value(self):
        self.assertAlmostEqual(0.5, scale_value(0, -1.0, 1.0, 0.0, 1.0))
        self.assertAlmostEqual(0.25, scale_value(0, -1.0, 1.0, 0.0, 1.0, 2))

    def test_contour_approx(self):
        for inputs, outputs in zip(self.TEST_INPUTS, self.TEST_OUTPUTS):
            self.assertTrue(
                np.array_equal(
                    sorted(
                        list(get_corners_from_contour(inputs, len(outputs))),
                        key=lambda x: x[0][0],
                    ),
                    sorted(list(outputs), key=lambda x: x[0][0]),
                )
            )

    def test_get_angles(self):
        self.assertAlmostEqual(
            math.radians(45), get_horizontal_angle(100, 100, math.radians(45))
        )
        self.assertAlmostEqual(
            math.radians(30),
            get_vertical_angle_linear(100, 400, math.radians(60), inverted=True),
        )
        self.assertAlmostEqual(
            math.radians(-30),
            get_vertical_angle_linear(300, 400, math.radians(60), inverted=True),
        )

    def test_get_distance(self):
        self.assertAlmostEqual(-1.0, get_distance(math.radians(-45), 2, 1, 0))
        self.assertAlmostEqual(1.0, get_distance(0, 3, 2, math.radians(45)))
        self.assertAlmostEqual(
            1.0, get_distance(math.radians(20), 3, 2, math.radians(25))
        )
        self.assertAlmostEqual(
            math.sqrt(3), get_distance(math.radians(10), 4, 3, math.radians(20))
        )

    def test_get_values_solvepnp(self):
        # TODO implement
        pass


if __name__ == "__main__":
    unittest.main()
