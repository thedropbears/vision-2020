from camera_manager import MockImageManager
from connection import DummyConnection
import numpy as np
import os
import cv2
import balls_vision
from magic_numbers import *

def generateData(cap=-1, images_path = "tests/balls", outputFile = "balls_data.npz"):
    import balls_vision

    folders = os.listdir(images_path)
    files = []
    labels = []
    for d in folders: 
        try:
            getPathNum(d) 
            print(f"found file path{d}")
        except:
            print(f"found non path file {d}")

        images = os.listdir(os.path.join(images_path, d))
        for i in images:
            files.append(os.path.join(d, i))
            labels.append(d)

    dataLabels = []
    dataPoss = []
    for n, i in enumerate(files):
        label = labels[n]
        img = cv2.imread(os.path.join(images_path, i))
        vision = balls_vision.Vision(MockImageManager(img), DummyConnection())
        res = np.array(vision.normalize(vision.find_balls(img)))
        print(f"found {res.shape[0]/2} balls")
        if res.shape[0] == 6:
            dataPoss.append(res)
            dataLabels.append(getPathNum(label))
            # print(res, label, getPathNum(label))
        else:
            print("didnt find three balls")
    dataPoss = np.array(dataPoss, dtype=np.int8)
    dataLabels = np.array(dataLabels, dtype=np.uint8)
    np.savez(outputFile, labels=dataLabels, balls=dataPoss)


if __name__ == "__main__":

    # gets the ball vision results form the images in ./tests/balls/ and saves them as balls_data.npz

    generateData()
