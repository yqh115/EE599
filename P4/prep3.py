import os
import shutil
import numpy as np
import pandas as pd

# copy images to labeled directory
for i in range(0, 8):
    os.mkdir('C:/Users/ME/PycharmProjects/EE599/HW4/data/train'+'/'+str(i))
    #os.mkdir('C:/Users/ME/PycharmProjects/EE599/HW4/data/val'+'/'+str(i))
