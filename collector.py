from balls_vision import Vision
from camera_manager import MockImageManager
from connection import DummyConnection
import numpy as np
import os
import cv2


pathLabels = {1: "A1", 2: "A2", 3: "B1", 4: "B2", 0: "None"}
getNum = lambda x: list(pathLabels.keys())[
    list(pathLabels.values()).index(x)
]  # gets the path num from its str name
getStr = lambda x: pathLabels[x]  # gets the path str name from its num

def generateData(cap=-1, images_path = "tests/balls", outputFile = "balls_data.npz"):
    files = os.listdir(images_path)[:cap]
    dataLabels = []
    dataPoss = []
    for i in files:
        label = i[:2]
        img = cv2.imread(os.path.join(images_path, i))
        vision = Vision(MockImageManager(img), DummyConnection())
        res = np.array(vision.find_balls(img))
        if res.shape[0] == 9:
            dataPoss.append(res)
            dataLabels.append(getNum(label))
            # print(res, label, getNum(label))
    dataPoss = np.array(dataPoss, dtype=np.int8)
    dataLabels = np.array(dataLabels, dtype=np.uint8)
    np.savez(outputFile, labels=dataLabels, balls=dataPoss)


if __name__ == "__main__":

    # gets the ball vision results form the images in ./tests/balls/ and saves them as balls_data.npz

    generateData()
