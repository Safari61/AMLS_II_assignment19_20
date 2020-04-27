import os
import cv2
import pickle
import random
import numpy as np

DATADIR = "E:/Machine Learning/AMLS2 project/dataset/train"

raw_data_150 = []

for img in os.listdir(DATADIR):
    names = img.split('.')[0]
    if names == 'cat':
        label = 0
    elif names == 'dog':
        label = 1
    pixels_150 = cv2.resize(cv2.imread(os.path.join(DATADIR, img), 0), (150, 150))
    raw_data_150.append([label, pixels_150])
#     print(raw_data_50)
#
# print("total:")
# print(raw_data_50)
random.shuffle(raw_data_150)
raw_data_150 = np.array(raw_data_150)
# print(len(raw_data_100))
# print(len(raw_data_50))
# print(raw_data_50)
# print(raw_data_50.shape)
# print(raw_data_50)


y_150 = []
X_150 = []
for l, p in raw_data_150:
    y_150.append(l)
    X_150.append(p)


# print(y_50)
pickle_out = open("X_150", "wb")
pickle.dump(X_150, pickle_out)
pickle_out.close()


pickle_out = open("y_150", "wb")
pickle.dump(y_150, pickle_out)
pickle_out.close()


print('finish')