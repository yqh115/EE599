import matplotlib.pyplot as plt
import h5py
import numpy as np
import json

f1 = h5py.File('mnist_network_params.hdf5', 'r')
f2 = h5py.File('mnist_testdata.hdf5', 'r')

# print('keys of the f1: ', list(f1.keys()))
# print('keys of the f2: ', list(f2.keys()))
W1_read = f1['W1']
W2_read = f1['W2']
W3_read = f1['W3']
b1_read = f1['b1']
b2_read = f1['b2']
b3_read = f1['b3']
print('dataset: ', W1_read)
print('dataset: ', W2_read)
print('dataset: ', W3_read)
print('dataset: ', b1_read)
print('dataset: ', b2_read)
print('dataset: ', b3_read)

xdata_read = f2['xdata']
ydata_read = f2['ydata']
# print('dataset: ', xdata_read)
# print('dataset: ', ydata_read)

W1 = W1_read[:]
W2 = W2_read[:]
W3 = W3_read[:]
b1 = b1_read[:]
b2 = b2_read[:]
b3 = b3_read[:]
xdata = xdata_read[:]
ydata = ydata_read[:]

f1.close()
f2.close()


# print(xdata[1].shape)

a1 = np.dot(xdata, np.transpose(W1)) + b1
# print(a1[1])
a1 = np.maximum(a1, 0)
# print(a1[1])

a2 = np.dot(a1, np.transpose(W2)) + b2
# print(a2[1])
a2 = np.maximum(a2, 0)
# print(a2[1])

a3 = np.dot(a2, np.transpose(W3)) + b3


def softmax1(x):
    N, D = x.shape
    softmax_x = np.zeros((N, D))
    softmax_sum = np.sum(np.exp(x), axis=1)
    for i in range(0, N):
        for j in range(0, D):
            softmax_x[i][j] = np.exp(x[i][j]) / softmax_sum[i]

    # print(softmax_x)
    return softmax_x

def softmax2(x):

    softmax_x = np.zeros((1, 10), float)
    softmax_sum = np.sum(np.exp(x), axis=1)

    for j in range(0, D):
        softmax_x[0, j] = np.exp(x[0, j]) / softmax_sum

    # print(softmax_x)
    return softmax_x
# print(softmax_sum.shape)


x = softmax1(a3)
maxx = np.argmax(x, axis=1)
# print(maxx.shape)
# print(ydata.shape)
N, D = ydata.shape
labels = np.zeros((N, D))

# for i in range(0, N):
#     j = maxx[i]
#     # print(j)
#     labels[i][j] = 1
#
# diff = labels - ydata
# # print(diff.shape)
# print(np.sum(abs(diff)))
# num_true = N - np.sum(abs(diff)) / 2
# print(num_true)

file_obj = open('prob1', 'w')
output = []
for idx in range(0, len(x)):
    # input = xdata[idx].reshape(1, 784)
    # a1 = np.dot(input, W1.T) + b1
    # # print(a1.shape)
    # # a1 = a1.reshape(1, 200)
    # a2 = np.dot(a1, W2.T) + b2
    # # a2 = a2.reshape(100, 1)
    # a3 = np.dot(a2, W3.T) + b3
    # # a3 = a3.reshape(10, 1)
    # o3 = softmax2(a3)
    # # print(o3.shape)
    output.append({
        "activations": x[idx, :].tolist(),
        "index": idx,
        "classification": int(x[idx, :].argmax())
    })
json.dump(output, file_obj)
# print(x)
# print(output)
print("AUTOGRADE: %s" % (json.dumps(output)))
