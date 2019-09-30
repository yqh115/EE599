import librosa
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential, Model
from keras import layers
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
import numpy as np
import os
from sklearn.model_selection import train_test_split

ENG_DIR = 'C:/Users/ME/Desktop/mfcctrain/train/english/'
HIN_DIR = 'C:/Users/ME/Desktop/mfcctrain/train/hindi/'
CHN_DIR = 'C:/Users/ME/Desktop/mfcctrain/train/mandarin/'


def load_voicedata(num_eng, num_chn, num_hin, length):
    eng_data = np.zeros((num_eng, 64, length), dtype=float)
    eng_labels = np.zeros((num_eng, 1), dtype=float)
    chn_data = np.zeros((num_chn, 64, length), dtype=float)
    chn_labels = np.ones((num_chn, 1), dtype=float) * 2
    hin_data = np.zeros((num_hin, 64, length), dtype=float)
    hin_labels = np.ones((num_hin, 1), dtype=float)
    eng_voice = [ENG_DIR + i for i in os.listdir(ENG_DIR)]
    chn_voice = [CHN_DIR + i for i in os.listdir(CHN_DIR)]
    hin_voice = [HIN_DIR + i for i in os.listdir(HIN_DIR)]
    for i, eng_file in enumerate(eng_voice):
        read_data = np.load(eng_file)
        eng_data[i] = read_data[:, :length]
        if i == num_eng - 1:
            break
    print('finished preparing english data')
    for i, chn_file in enumerate(chn_voice):
        read_data = np.load(chn_file)
        chn_data[i] = read_data[:, :length]
        if i == num_chn - 1:
            break
    print('finished preparing madarin data')
    for i, hin_file in enumerate(hin_voice):
        read_data = np.load(hin_file)
        hin_data[i] = read_data[:, :length]
        if i == num_hin - 1:
            break
    print('finished preparing hindi data')
    data = np.concatenate((eng_data, chn_data, hin_data), axis=0)
    labels = np.concatenate((eng_labels, chn_labels, hin_labels), axis=0)
    print('finished preparing data')
    print('data size', data.shape)
    print(labels)
    return data, labels

def data_preprocessing(input_data, input_labels):
    data_mean_mid = input_data.mean(axis=0)
    data_mean = data_mean_mid.mean(axis=1)
    data_std_mid = input_data.std(axis=0)
    data_std = data_std_mid.std(axis=1)
    output_data = input_data
    for i in range(0, len(input_data)):
        for j in range(0, input_data.shape[2]):
            output_data[i, :, j] -= data_mean
            output_data[i, :, j] = output_data[i, :, j] / data_std
    #    output_data=(input_data-data_mean)/data_std
    output_labels = np.zeros((len(input_data), 3), dtype=float)
    for i in range(0, len(input_data)):
        if input_labels[i] == 0:
            output_labels[i, 0] = 1
        elif input_labels[i] == 1:
            output_labels[i, 1] = 1
        else:
            output_labels[i, 2] = 1
    print('finish data preprocessing, get standard data and one hot labels')
    print(output_labels)
    return output_data, output_labels

def data_generator(input_data, input_labels, each_length, gene_times):
    gene_data = np.zeros((len(input_data) * gene_times, 64, each_length), dtype=float)
    gene_labels = np.zeros((len(input_data) * gene_times, 3), dtype=float)
    for i in range(0, len(input_data)):
        for j in range(0, gene_times):
            gene_data[i * gene_times + j] = input_data[i, :,
                                            j * (each_length + 100):j * (each_length + 100) + each_length]
            gene_labels[i * gene_times + j] = input_labels[i]

    return gene_data, gene_labels

def model_GRU():
    model = Sequential()
    model.add(layers.GRU(64,
                         dropout=0.1,
                         recurrent_dropout=0.6,
                         return_sequences=True,
                         input_shape=(600, gene_data.shape[-1])))
    # model.add(layers.Dense(64, activation='relu'))
    model.add(layers.GRU(32,
                         dropout=0.1,
                         recurrent_dropout=0.5,
                         return_sequences=True,
                         input_shape=(600, gene_data.shape[-1])))
    # model.add(layers.Dense(32, activation='relu'))
    model.add(layers.GRU(16,
                         dropout=0.1,
                         recurrent_dropout=0.4,
                         return_sequences=True,
                         input_shape=(32, gene_data.shape[-1])))
    model.add(layers.Flatten())
    model.add(layers.Dense(8, activation='relu'))
    #    model.add(layers.Flatten())
    # model.add(layers.Dense(32, activation='relu'))
    # model.add(layers.Dense(8, activation='relu'))
    model.add(layers.Dense(3, activation='softmax'))
    #    model.add(Activation('sigmoid'))
    return model


# def show_plot():
#     loss = history.history['loss']
#     val_loss = history.history['val_loss']
#
#     epochs = range(len(loss))
#
#     plt.figure()
#
#     plt.plot(epochs, loss, 'bo', label='Training loss')
#     plt.plot(epochs, val_loss, 'b', label='Validation loss')
#     plt.title('Training and validation loss')
#     plt.legend()
#
#     plt.show()

#############################################################################################################
# main function
data, labels = load_voicedata(60, 60, 60, 20000)
all_data, all_labels = data_preprocessing(data, labels)
gene_data, gene_labels = data_generator(all_data, all_labels, each_length=600, gene_times=25)
gene_data = np.reshape(gene_data, (len(gene_data), 600, 64))
train_data, val_data, train_labels, val_labels = train_test_split(gene_data, gene_labels, test_size=0.1666)
print('train size:', train_data.shape, 'label size:', train_labels.shape)

model = model_GRU()
model.summary()

n_batch = 32
n_epoch = 10
# train_data_new=train_data.reshape(1200,1,1,64, 600)
train_labels_re = np.reshape(train_labels, (train_labels.shape[0], 1, 3))
print(train_data.shape)
print(train_labels.shape)
train_data_re = np.reshape(train_data, (train_data.shape[0], train_data.shape[1], train_data.shape[2]))
print('input data shape:', train_data_re.shape, 'input label shape:', train_labels_re.shape, 'batch size', n_batch)
model.compile(optimizer=RMSprop(), loss='categorical_crossentropy')

early_stopping = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')

history = model.fit(train_data_re, train_labels, epochs=n_epoch, batch_size=n_batch,
                    validation_split=0.2, callbacks=[early_stopping])

val_data = np.reshape(val_data, (val_data.shape[0], 600, 64))
prediction = model.predict(val_data, verbose=0)
prediction = prediction.mean(axis=1)
model.save('hw5_GRU_10.h5')
#show_plot()
