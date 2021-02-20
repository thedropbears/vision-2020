from balls_vision import Vision
from camera_manager import MockImageManager
import numpy as np
import os
import cv2


images_path = "tests/balls"
pathLabels = {1: "A1", 2: "A2", 3: "B1", 4: "B2", 0: "None"}
getNum = lambda x: list(pathLabels.keys())[
    list(pathLabels.values()).index(x)
]  # gets the path num from its str name
getStr = lambda x: pathLabels[x]  # gets the path str name from its num

if __name__ == "__main__":

    # gets the ball vision results form the images in ./tests/balls/ and saves them as balls_data.npz

    files = os.listdir(images_path)
    dataLabels = []
    dataPoss = []
    for i in files:
        label = i[:2]
        img = cv2.imread(os.path.join(images_path, i))
        vision = Vision(MockImageManager(img))
        res = vision.find_balls()
        dataPoss.append(res)
        dataLabels.append(getNum(label))
        print(res, labels, getNum(label))
        while len(dataPoss[-1]) < 3:  # create null balls if it didnt see three
            dataPoss[-1].append([0, 0, 0])

    dataPoss = np.array(dataPoss, dtype=np.float32)
    dataLabels = np.array(dataLabels, dtype=np.uint8)
    with open("balls_data.npz", "wb") as f:
        np.savez(f, labels=dataLabels, balls=dataPoss)
