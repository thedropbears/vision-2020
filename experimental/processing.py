from magic_numbers import *
import cv2
import time
import numpy as np


def pixel_green_check(pixel):  # (B, G, R)
    """Tests if a pixel is 'green enough'"""
    if pixel[1] <= MIN_GREEN:
        return False
    if not pixel[1] > pixel[0] > pixel[2]:
        return False
    chroma = pixel[1] - pixel[2]
    if (chroma << 8) // pixel[1] <= MIN_SATURATION:
        return False
    if not H_MAX > ((pixel[0] - pixel[2]) << 8) // chroma > H_MIN:
        return False
    return True


def test_homemade():
    img = cv2.imread("green.jpg")
    mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    t = time.time()
    for i, row in enumerate(img):
        for j, p in enumerate(row):
            if pixel_green_check(p):
                mask[i][j] = 255
            else:
                mask[i][j] = 0
    print(time.time() - t)
    return mask


def test_cv():
    img = cv2.imread("green.jpg")
    mask1 = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    hsv = np.zeros(img.shape, np.uint8)
    t = time.time()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV, dst=hsv)
    mask1 = cv2.inRange(hsv, (75, 128, 128), (85, 255, 255), dst=mask1)
    print(time.time() - t)
    return mask1


if __name__ == "__main__":
    mask = test_homemade()
    mask1 = test_cv()
    cv2.imshow("mask", mask)
    cv2.imwrite("mask.jpg", mask)
    cv2.imshow("mask1", mask1)
    cv2.imwrite("cvmask.jpg", mask1)
    cv2.waitKey(0)
