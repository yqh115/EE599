import h5py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

import keras
from keras.models import Model
from keras.layers import GRU
from keras.layers import Input
from keras.layers import Dense
# from keras import metrics
from keras import losses

BATCH_SIZE = 32
EPOCHS = 40

USE_REAL_DATA = True


def generate_fake_data(N, L):
    y = np.zeros((N, L))
    for n in range(N):
        for k in range(L):
            if y[n][k - 1] == 1 and y[n][k - 1] == 1 and y[n][k - 2] == 1:
                if np.random.binomial(1, 0.8):
                    y[n][k] = 0
                else:
                    y[n][k] = np.random.binomial(1, 0.5)
            elif y[n][k - 1] == 0 and y[n][k - 1] == 0 and y[n][k - 2] == 0:
                if np.random.binomial(1, 0.8):
                    y[n][k] = 1
                else:
                    y[n][k] = np.random.binomial(1, 0.5)
    return y


############################################################################################
#############################  Get the Data/Format ##########################################
############################################################################################

if USE_REAL_DATA:
    IN_FNAME = '/Volumes/GoogleDrive/Team Drives/EE599_Sp2019_teaching/data.usc-ece.com-sync/HW2/5/binary_random_sp2019.hdf5'
    with h5py.File(IN_FNAME, 'r') as hf:
        human = hf['human'][:]
else:
    human = generate_fake_data(1000, 20)

N_sequences = human.shape[0]
L_sequence = human.shape[1]

target_sequences = np.copy(np.roll(human, -1, axis=1))

#### reshape the data to be a N_sequences of length L_sequence with feature size 1

human = np.reshape(human, (N_sequences, L_sequence, 1))
target_sequences = np.reshape(target_sequences, (N_sequences, L_sequence, 1))

############################################################################################
#############################  Define/Build Model ##########################################
############################################################################################

print('\nBuilding model...')

DROPOUT = 0.0
RECURRENT_DROP_OUT = 0.0

GRU_nodes = 4
main_input = Input(shape=(None, 1), name='main_input')
pred_gru = GRU(GRU_nodes, return_sequences=True, dropout=DROPOUT, recurrent_dropout=RECURRENT_DROP_OUT,
               name='pred_gru')(main_input)
rnn_output = Dense(1, activation='sigmoid', name='rnn_output')(pred_gru)

model = Model(inputs=main_input, outputs=rnn_output)
print('\nCompiling model...')
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.summary()

############################################################################################
#############################  Train and saver model #######################################
############################################################################################
print('\nTraining...')
history = model.fit(human, target_sequences, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.2, shuffle=1)
model.save('mind_reader_sequence.hdf5')

fig = plt.figure()
plt.plot(history.history['acc'], label='train')
plt.plot(history.history['val_acc'], label='val')
plt.title('Traing Accuracy')
axes = plt.gca()
axes.set_ylim([0, 1])
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.legend()

# fig = plt.figure()
# plt.plot( history.history['loss'], label = 'train')
# plt.plot( history.history['val_loss'], label = 'val')
# plt.title('Cross Entropy Loss')
# axes = plt.gca()
# plt.ylabel('BCE Entropy')
# plt.xlabel('epochs')

############################################################################################
#############################  Do Inference Example ##########################################
############################################################################################
streaming_input = Input(name='streaming_input', batch_shape=(1, 1, 1))
pred_gru = GRU(GRU_nodes, name='pred_gru', stateful=True)(streaming_input)
rnn_output = Dense(1, name='rnn_output')(pred_gru)

streaming_model = Model(inputs=streaming_input, outputs=rnn_output)
streaming_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['acc'])

old_weights = model.get_weights()
streaming_model.set_weights(old_weights)

one_sequence = generate_fake_data(1, L_sequence)
one_sequence = np.reshape(one_sequence, (1, L_sequence, 1))
correct = np.copy(np.roll(one_sequence, -1, axis=1))

### predict using the sequence to sequence (trained) model:
seq_predictions = model.predict(np.reshape(one_sequence, (1, 20, 1)))

### predict using the sequence to sequence (trained) model:
stream_predictions = np.zeros(L_sequence)

n_correct = 0
for k in range(20):
    sample = np.ones((1, 1, 1)) * one_sequence[0][k]
    stream_predictions[k] = streaming_model.predict(sample)
    if int(stream_predictions[k] > 0.5) == int(correct[0][k]):
        corr_txt = '(*)'
        n_correct += 1
    else:
        corr_txt = ''
    print('k= ', k, 'input, correct, stream, seq: ', int(sample[0][0][0]), int(correct[0][k][0]),
          int(stream_predictions[k] > 0.5), int(seq_predictions[0][k][0] > 0.5), corr_txt)
    print('k= ', k, 'stream, seq: ', stream_predictions[k], seq_predictions[0][k][0])

print('Percent Correct: ', 100.0 * n_correct / L_sequence)
## use this if you want to start another sequence of calls to the streaming model
streaming_model.reset_states()

fig = plt.figure()
plt.stem(correct[0], label='correct')
plt.stem(np.asarray(seq_predictions[0].T[0] > 0.5), linefmt='tab:gray', markerfmt='ro', label='pred')
plt.title('Correct and Guesses')
axes = plt.gca()
axes.set_ylim([0, 1.2])
plt.ylabel('0 or 1 value')
plt.xlabel('sequence position')
plt.legend()
