############################################################
# Author: Sourya Dey
# Spring 2019, USC EE599 Deep Learning
###########################################################

import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.initializers import Constant
from keras.regularizers import l1, l2
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
# Define hyperparameters
# =============================================================================
lamda = 1e-5 #cannot use lambda as variable name
biasinit = 0.1
wtinit = 'he_normal'
optimizer = Adam()
loss = 'categorical_crossentropy'
metrics = ['accuracy']
batch_size = 256
epochs = 10

# =============================================================================
# Define and train a sequential model in Keras
# =============================================================================
model = Sequential()
#in the next line, change kernel_regularizer to l1, l2, l4 to get different results
model.add(Dense(100, input_shape=(784,), activation='relu', kernel_initializer=wtinit, bias_initializer=Constant(biasinit), kernel_regularizer=l1(lamda)))
model.add(Dense(10, activation='softmax', kernel_initializer=wtinit, bias_initializer=Constant(biasinit)))
model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
print(model.summary())
history = model.fit(xtr,ytr, batch_size=batch_size, epochs=epochs, validation_data=(xva,yva), verbose=False)

# =============================================================================
# Test
# =============================================================================
score = model.evaluate(xte,yte) #outputs [loss,accuracy]
print('Test accuracy after {0} epochs = {1}%\n\n'.format(epochs,100*score[1]))


# =============================================================================
# Analyze weights
# =============================================================================
print(len(model.get_weights()))
print(model.get_weights()[0].shape)

jn1w = model.get_weights()[0].flatten()
print('Max weight value = {0}'.format(np.max(jn1w)))
print('Min weight value = {0}'.format(np.min(jn1w)))

plt.figure(figsize=(8,8))
plt.hist(jn1w, weights = np.ones_like(jn1w)/float(len(jn1w)), bins=50)
plt.grid()
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.show()
