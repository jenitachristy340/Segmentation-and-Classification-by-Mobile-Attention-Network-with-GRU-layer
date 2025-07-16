# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
from Evaluation import evaluation
plt.style.use('fivethirtyeight')
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
import math
from sklearn.metrics import mean_squared_error



def Model_GRU(X_train, y_train, X_test, Y_test, Optimizer):
    # The GRU architecture
    regressorGRU = Sequential()
    # First GRU layer with Dropout regularisation
    regressorGRU.add(GRU(units=10, return_sequences=True, input_shape=(X_train.shape[1],1), activation='tanh'))
    regressorGRU.add(Dropout(0.2))
    # Second GRU layer
    regressorGRU.add(GRU(units=10, return_sequences=True, input_shape=(X_train.shape[1],1), activation='tanh'))
    regressorGRU.add(Dropout(0.2))
    # Third GRU layer
    regressorGRU.add(GRU(units=10, return_sequences=True, input_shape=(X_train.shape[1],1), activation='tanh'))
    regressorGRU.add(Dropout(0.2))
    # Fourth GRU layer
    regressorGRU.add(GRU(units=10, activation='tanh'))
    regressorGRU.add(Dropout(0.2))
    # The output layer
    regressorGRU.add(Dense(units=1))
    # Compiling the RNN
    regressorGRU.compile(optimizer='Adam',loss='mean_squared_error') # SGD(lr=0.01, decay=1e-7, momentum=0.9, nesterov=False)
    # Fitting to the training set
    trx_data = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    tex_data = X_test.reshape((X_test.shape[0], X_test[1].shape[0], 1))
    # tey_data = y_train.reshape((y_train.shape[0], y_train.shape[1]))
    pred = np.zeros((Y_test.shape))
    for i in range(y_train.shape[1]): # for all classes
        regressorGRU.fit(trx_data, y_train[:, i], epochs=1, batch_size=150)
        pr = regressorGRU.predict(tex_data).ravel()
        #pr = regressorGRU.predict(tex_data)
        for j in range(pr.shape[0]):
            # pred[j, i] = np.mean(pr[j, :, :])
            pred[j, i] = np.mean(pr[j])
            if np.isnan(pred[j, i]):
                pred[j, i] = 0
    Eval = evaluation(pred, Y_test)
    return Eval, pred

