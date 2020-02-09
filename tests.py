import unittest
import cv2
import numpy as np
from vision import Vision
from utilities.functions import get_corners_from_contour


class VisionTests(unittest.TestCase):
    def test_sample_images(self):
        f = open("./tests/results.csv", "r")
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
                [[152, 79]],
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


if __name__ == "__main__":
    camera_server = Vision(test=True)
    unittest.main()
