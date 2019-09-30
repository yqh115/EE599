import csv
import numpy as np
import pandas as pd

#file = open('C:/Users/ME/PycharmProjects/EE599/HW4/data.csv', 'r')
reader = pd.read_csv('C:/Users/ME/PycharmProjects/EE599/HW4/data.csv', chunksize=3248)
# all_data = csv.reader(file)
# loop = True
# chunkSize = 5000
# chunks = []
# while loop:
#     try:
#         chunk = reader.get_chunk(chunkSize)
#         chunks.append(chunk)
#     except StopIteration:
#         loop = False
#         print("Iteration is stopped.")
# df = pd.concat(chunks, ignore_index=True)

# print(all_data.shape)
train_data = []
val_data = []
test_data = []
i = 0
f1 = open("C:/Users/ME/PycharmProjects/EE599/HW4/train_data.csv", "w", newline='')
f2 = open("C:/Users/ME/PycharmProjects/EE599/HW4/val_data.csv", "w", newline='')
f3 = open("C:/Users/ME/PycharmProjects/EE599/HW4/test_data.csv", "w", newline='')
for ck in reader:
    if (i == 0 or i == 1):
        train_data = pd.DataFrame(ck)
        train_data.to_csv("train_data.csv", mode='a', header=False)

    elif i == 2:

        val_data = pd.DataFrame(ck)
        val_data.to_csv("val_data.csv", mode='a', header=False)
        #i = i + 1
    else:

        test_data = pd.DataFrame(ck)
        test_data.to_csv("test_data.csv", mode='a', header=False)
        #i = i + 1
    i = i + 1
# file.close()
# f1 = open("C:/Users/ME/PycharmProjects/EE599/HW4/train_data.csv", "w", newline='')
# f2 = open("C:/Users/ME/PycharmProjects/EE599/HW4/val_data.csv", "w", newline='')
# f3 = open("C:/Users/ME/PycharmProjects/EE599/HW4/test_data.csv", "w", newline='')
# writer1 = csv.writer(f1)
# writer1.writerows(train_data)
# f1.close()
# writer2 = csv.writer(f2)
# writer2.writerows(val_data)
# f2.close()
# writer3 = csv.writer(f3)
# writer3.writerows(test_data)
# f3.close()
