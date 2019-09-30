############################################################
# Author: Sourya Dey
# Helped by: Arindam Jati
# Spring 2019, USC EE599 Deep Learning
###########################################################

import numpy as np
import keras.backend as K

from keras.models import Sequential
from keras.layers import Dense
from keras.initializers import Constant
from keras.regularizers import l2
#from keras.models import load_model
from keras.optimizers import Adam


# =============================================================================
# Load data
# =============================================================================
loaded = np.load('./mnist.npz')
xtr = loaded['xtr']
ytr = loaded['ytr']
xva = loaded['xva']
yva = loaded['yva']
xte = loaded['xte']
yte = loaded['yte']

# =============================================================================
# Define custom loss function
# =============================================================================
def euclidean_distance_loss(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))

# =============================================================================
# Define hyperparameters
# =============================================================================
lamda = 1e-5
biasinit = 0.1
wtinit = 'he_normal'
optimizer = Adam()
#loss = 'categorical_crossentropy'
loss = euclidean_distance_loss
metrics = ['accuracy']
batch_size = 256
epochs = 10

# =============================================================================
# Define and train a sequential model in Keras
# =============================================================================
model = Sequential()
model.add(Dense(100, input_shape=(784,), activation='relu', kernel_initializer=wtinit, bias_initializer=Constant(biasinit), kernel_regularizer=l2(lamda)))
model.add(Dense(10, activation='softmax', kernel_initializer=wtinit, bias_initializer=Constant(biasinit), kernel_regularizer=l2(lamda)))
model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
print(model.summary())
history = model.fit(xtr,ytr, batch_size=batch_size, epochs=epochs, validation_data=(xva,yva))

# =============================================================================
# Test
# =============================================================================
score = model.evaluate(xte,yte) #outputs [loss,accuracy]
print('Test accuracy after {0} epochs = {1}%'.format(epochs,100*score[1]))

# =============================================================================
# Save model
# =============================================================================
model.save('mymodel.h5')

#### Do not load model or Keras will give an error about undefined custom loss function
#### Instead use the permanent fix given in the Piazza thread 'Fix to Keras custom loss problem'
