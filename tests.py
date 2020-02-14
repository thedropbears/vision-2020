import unittest
import cv2
import numpy as np
from vision import Vision
from utilities.functions import *


class VisionTests(unittest.TestCase):
    def test_sample_images(self):
        f = open("./tests/power_port/results.csv", "r")
        lines = f.read().split("\n")
        f.close()
        for line in lines[1:]:
            values = line.split(",")  # Filename.jpg, 1, 2
            results = camera_server.get_image_values(cv2.imread(f"./tests/{values[0]}"))
            if results is not None:
                for i in range(1, len(values)):
                    self.assertAlmostEqual(results[i - 1], float(values[i]))

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
            [[[67, 40]], [[258, 43]], [[211, 160]], [[86, 146]]],
            [
                [[66, 65]],
                [[116, 50]],
                [[152, 79]],
                [[152, 121]],
                [[100, 132]],
                [[64, 116]],
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

    def test_get_distance(self):
        self.assertAlmostEqual(1.0, get_distance(math.radians(-45), 2, 1, 0))
        self.assertAlmostEqual(1.0, get_distance(0, 3, 2, math.radians(45)))
        self.assertAlmostEqual(
            1.0, get_distance(math.radians(-20), 3, 2, math.radians(25))
        )
        self.assertAlmostEqual(
            math.sqrt(3), get_distance(math.radians(-10), 4, 3, math.radians(20))
        )

    def test_get_angles(self):
        self.assertAlmostEqual(
            math.radians(45), get_horizontal_angle(200, np.array([[100.0, 0.0, 100.0]]))
        )
        self.assertAlmostEqual(
            math.radians(30),
            get_vertical_angle(100.0, np.array([[], [0.0, 50 * math.sqrt(3), 50.0]])),
        )


if __name__ == "__main__":
    unittest.main()
