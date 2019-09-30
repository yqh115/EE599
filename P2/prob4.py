import matplotlib.pyplot as plt
import h5py
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle
import json

f1 = h5py.File('lms_fun_v2.hdf5', 'r')
f2 = h5py.File('lms_fun_v3.hdf5', 'r')

#print('keys of the f1: ', list(f1.keys()))
#print('keys of the f2: ', list(f2.keys()))

mismatched_v2_read = f1['mismatched_v']
mismatched_v3_read = f2['mismatched_v']
mismatched_y2_read = f1['mismatched_y']
mismatched_y3_read = f2['mismatched_y']

v2 = mismatched_v2_read[:].reshape((300600, 3))
v3 = mismatched_v3_read[:].reshape((300600, 3))
y2 = mismatched_y2_read[:].reshape((300600, 1))
y3 = mismatched_y3_read[:].reshape((300600, 1))
#print(v2.shape)
#print(y3.shape)

V2_train, V2_test, y2_train, y2_test = train_test_split(v2, y2, test_size=0.9)
V3_train, V3_test, y3_train, y3_test = train_test_split(v3, y3, test_size=0.9)

# print(V2_train.shape)
# print(V3_train.shape)
# print(V2_test.shape)
# print(V3_test.shape)
# print(y2_train.shape)
# print(y2_test.shape)

# for i in range(1, 10):
#     for j in range(1, 10):
#         if i+j >10:
#             break
#         nn = MLPRegressor(hidden_layer_sizes=(i, j), solver='lbfgs', activation='relu', alpha=1e-4, random_state=1)
#         nn.fit(V2_train, y2_train)
#         y2_predict = nn.predict(V2_test)
#
#         MSE = mean_squared_error(y2_predict, y2_test)
#         print('(', i, ', ', j, ')', '    ')
#         print(mean_squared_error(y2_predict, y2_test))


nn = MLPRegressor(hidden_layer_sizes=(5, 3), solver='lbfgs', activation='relu', alpha=1e-4, random_state=1)
nn.fit(V3_train, y3_train)
y3_predict = nn.predict(V3_test)

MSE = mean_squared_error(y3_predict, y3_test)
print(MSE)
pickle_file = open('nn.pkl', 'wb')
pickle.dump(nn, pickle_file)
pickle_file.close()

