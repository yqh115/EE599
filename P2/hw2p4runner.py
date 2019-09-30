import h5py
import pickle
import json

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

dataFile = './hw2p3-test-data.hdf5'

with h5py.File(dataFile, 'r+') as df:
    X = df['v'][:]
    y = df['y'][:]

    X = X.reshape((-1,3))
    y = y.reshape((-1,))

    nn = pickle.load(open('nn.pkl', 'rb'))

    y_pred = nn.predict(X)
    mse = mean_squared_error(y, y_pred)

    print('MSE: %.8f' % (mse))

    nneurons = sum(nn.get_params()['hidden_layer_sizes'])

    output = {
        "scores": {
            "Neurons": nneurons,
            "MSE": mse
        }
    }
    print(json.dumps(output))
