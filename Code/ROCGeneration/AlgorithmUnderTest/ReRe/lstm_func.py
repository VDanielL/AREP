"""
LSTM FUNCTIONS

Designed for the RePAD and ReRe algorithms and subsequent improvements like Alter-ReRe and AREP.
"""

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


class Lstm:
    def __init__(self, b, num_neurons):
        self.model = Sequential()
        self.model.add(LSTM(units=num_neurons, return_sequences=False, batch_input_shape=(1, b - 1, 1), stateful=True))
        self.model.add(Dense(1))
        self.model.compile(loss='mse', optimizer='adam')

    def train_lstm(self, train_data, num_epochs, debug):
        # splitting and reshaping data:
        x_train = train_data[:-1]
        y_train = train_data[-1]
        x_train = x_train.reshape(1, x_train.shape[0], 1)
        y_train = y_train.reshape(1)

        for count in range(num_epochs):
            self.model.fit(x_train,
                           y_train,
                           batch_size=1,
                           epochs=1,
                           verbose=0,
                           shuffle=False)
            self.model.reset_states()
        if debug:
            print('\t\tLSTM training done.')

    def predict_lstm(self, prev_data, time, debug, total_length, next_val):
        prev_data = prev_data.reshape(1, prev_data.shape[0], 1)
        pred = self.model.predict(prev_data, verbose=0)
        if debug:
            print('\t\tNew prediction made, value: {}'.format(round(float(pred[0, 0]), 5)))
            if time < total_length - 1:
                print('\t\t(Actual value: {})'.format(round(float(next_val), 5)))
        return pred[0, 0]

    def set_weights(self, lstm):
        self.model.set_weights(lstm.model.get_weights())
