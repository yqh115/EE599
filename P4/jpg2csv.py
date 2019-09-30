import csv, os, cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def convert_img_to_csv(img_dir):
    # 设置需要保存的csv路径
    with open(r"C:/Users/ME/PycharmProjects/EE599/HW4/data2.csv", "w", newline="") as f:
        # 设置csv文件的列名
        column_name = ["image", "emotion"]
        column_name.extend(["pixel%d" % i for i in range(128 * 128)])
        # 将列名写入到csv文件中
        writer = csv.writer(f)
        writer.writerow(column_name)

        # 获取目录的路径
        img_temp_dir = os.path.join(img_dir, "train_image")
        # 获取该目录下所有的文件
        img_list = os.listdir(img_temp_dir)
        csv_data2 = pd.read_csv('C:/Users/ME/PycharmProjects/EE599/HW4/train.csv', header=None)
        csv_df = pd.DataFrame(csv_data2)
        df = np.array(csv_df)

        #print(df)
        #csv_data2.columns = ['image', 'emotion']
        i = 0
        # 遍历所有的文件名称
        for img_name in img_list:
            # 判断文件是否为目录,如果为目录则不处理
            #if i<2:
                if not os.path.isdir(img_name):
                    # 获取图片的路径
                    img_path = os.path.join(img_temp_dir, img_name)
                    # 因为图片是黑白的，所以以灰色读取图片
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
                    # plt.figure("test")
                    # plt.imshow(img, cmap="gray")
                    # plt.show()
                    #print(img)
                    # 图片标签
                    row_data = []
                    # 获取图片的像素
                    row_data.extend(df[i])
                    #print(np.array(img.flatten()))
                    row_data.extend(img.flatten())
                    #print(row_data)
                    #data_i = pd.concat([csv_data2[i], row_data], axis=1)
                    # 将图片数据写入到csv文件中
                    writer.writerow(row_data)
                    print(i)
                    i = i + 1


if __name__ == "__main__":
    # 将该目录下的图片保存为csv文件
    convert_img_to_csv(r"C:\Users\ME\Desktop\train")
