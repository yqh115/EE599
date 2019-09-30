from keras.models import Sequential
from keras.layers import Dense, Activation, Input
from keras.models import Model
import h5py
import sys,pdb
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

############################################################
# Data: Load a subset of MNIST data
# split in into train and test
############################################################
# Load hdf5 file in 'r' mode
f = h5py.File('binary_random_sp2019.hdf5', 'r') #small dataset, assume this our full data
print('keys of the dataset: ', list(f.keys())) #useful for seeing the data set name/key
human_data = f['human'][:]
machine_data = f['machine'][:]
# print(human_data.shape, machine_data.shape)
human_label = np.ones((2400, 1))
machine_label = np.zeros((2400, 1))
human = np.concatenate((human_data, human_label), axis=1)
machine = np.concatenate((machine_data, machine_label), axis=1)
all_data = np.concatenate((machine, human), axis=0)
data = shuffle(all_data)
print(data[0:20])

split=3840
x_train = data[:split, 0:20]
x_test = data[split:, 0:20]
y_train = data[:split, 20]
y_test = data[split:, 20]
# print(x_train[0], 'x samples')
# print(y_test[0], 'y samples')

############################################################
# Define a functional API model
# Layer 1:                  20 * 20
# Activation 1:             relu
# Layer 2:                  20 * 1
# Activation 2:             sigmoid
############################################################
# This returns a tensor
inputs = Input(shape=(20,))

model = Sequential([
    Dense(100, input_shape=(20,)),
    Activation('relu'),
    Dense(20),
    Activation('relu'),
    Dense(1),
    Activation('sigmoid'),
])


# a layer instance is callable on a tensor, and returns a tensor
# x = Dense(32, activation='relu')(inputs)
# x = Dense(64, activation='relu')(x)
# predictions = Dense(10, activation='softmax')(x)

# This creates a model that includes
# the Input layer and three Dense layers
# model = Model(inputs=inputs, outputs=predictions)

############################################################
# Model printing
############################################################
# print(model.summary())


############################################################
# Model compilation
# Must be called before starting training
############################################################
# For a multi-class classification problem
model.compile(optimizer='sgd',
              loss='binary_crossentropy',
              metrics=['accuracy'])

############################################################
# Model training
############################################################
# if y_train is not categorical, you may use this
# or, write your own code to convert
#one_hot_labels = keras.utils.to_categorical(labels, num_classes=10)

history = model.fit(x_train, y_train, epochs=500, batch_size=32, validation_data=(x_test, y_test))


############################################################
# Model evaluation
############################################################
score = model.evaluate(x_test, y_test, batch_size=32)
print('Test Loss = ', score[0])
print('Test Accuracy = ', score[1])

history_dict = history.history
history_dict.keys()
#dict_keys(['val_loss', 'val_acc', 'loss', 'acc'])
epochs = range(500)
plt.clf()
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc_values, 'b', label='Training acc')
plt.plot(epochs, val_acc_values, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend()
plt.show()
model.save('my_model.hdf5')