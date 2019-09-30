import os
import numpy as np
import cv2 as cv

folder_path = "C:/Users/ME/Desktop/net_7876/"
for filename in os.listdir(folder_path):
    image = cv.imread(os.path.join(folder_path, filename))
    if image is None:
        print(filename)
    else:
        res = cv.resize(image, (160, 216), interpolation=cv.INTER_CUBIC)
        cv.imwrite(os.path.join("C:/Users/ME/Desktop/net_7876/net_7876/", filename), res)
        #print(image.shape)