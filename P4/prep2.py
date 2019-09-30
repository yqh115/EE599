import os
import shutil
import numpy as np
import pandas as pd
import cv2


name_and_label = []
name = []
label = []
reader = pd.read_csv('C:/Users/ME/PycharmProjects/EE599/HW4/train.csv', header=None)
list = pd.DataFrame(np.array(reader), columns=['name', 'label'])
name = list['name']
label = list['label']

c = pd.Categorical(label)
category = c.codes
print(c)
print(c.codes)
# copy images to labeled directory
# for i in range(0, 8):
#     os.mkdir('C:/Users/ME/PycharmProjects/EE599/HW4/data/all'+'/'+str(i))
# imgs = os.listdir('C:/Users/ME/PycharmProjects/EE599/HW4/train_image')
# imgnum = len(imgs)
# for i in range(imgnum):
#     label_i = category[i]
#     img_path = 'C:/Users/ME/PycharmProjects/EE599/HW4/train_image'+'/'+str(imgs[i])
#     img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#     img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_LINEAR)
#     cv2.imwrite('C:/Users/ME/PycharmProjects/EE599/HW4/data/all'+'/'+str(label_i)+'/'+str(imgs[i]), img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    # shutil.copy('C:/Users/ME/PycharmProjects/EE599/HW4/train/train_image'+'/'+str(imgs[i]),
    #             'C:/Users/ME/PycharmProjects/EE599/HW4/data/all'+'/'+str(label_i)+'/'+str(imgs[i]))