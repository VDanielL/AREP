"""
OFFSET COMPENSATION COMPONENT

This file contains the variables and functions necessary for offset compensation.
"""

import numpy as np


def init_offset_compensation(self):
    if self.USE_OFFSET_COMP:
        self.diff_avg_1 = list()
        self.diff_avg_2 = list()
        self.values_mean = list()
        self.pred_mean_1 = list()
        self.pred_mean_2 = list()
        self.offset_retrain_threshold = list()
        self.offset_retrain_trigger_1 = list()
        self.offset_retrain_trigger_2 = list()
        self.offset_retrain_signal_1 = list()
        self.offset_retrain_signal_2 = list()
        self.offset_ws_actual = 0
        self.offset_window_beg = 0
        self.offset_percentage_curr_1 = self.OFFSET_PERCENTAGE
        self.offset_percentage_curr_2 = self.OFFSET_PERCENTAGE

        # the first 2B-2 indices won't be used for these (first used at t = 2B - 1)
        for i in range(0, 2 * self.B - 1):
            if self.USE_OFFSET_COMP:
                self.diff_avg_1.insert(i, np.NaN)
                self.diff_avg_2.insert(i, np.NaN)
                self.values_mean.insert(i, np.NaN)
                self.pred_mean_1.insert(i, np.NaN)
                self.pred_mean_2.insert(i, np.NaN)
                self.offset_retrain_threshold.insert(i, np.NaN)
                self.offset_retrain_trigger_1.insert(i, np.NaN)
                self.offset_retrain_trigger_2.insert(i, np.NaN)
                self.offset_retrain_signal_1.insert(i, np.NaN)
                self.offset_retrain_signal_2.insert(i, np.NaN)

# criteria for triggering a retrain
def check_signal_in_ow(ow, time, signal):
    for i in range(ow, time + 1):
        if signal[i]:
            return False
    return True


def compensate_offset(self, time):
    # offset compensation component
    if self.USE_OFFSET_COMP and time >= 2 * self.B - 1:
        # calculating offset window size and beginning
        window_beg = time - self.WINDOW_SIZE if time - self.WINDOW_SIZE > self.B else self.B
        self.offset_ws_actual = self.OFFSET_WINDOW_SIZE if not self.USE_WINDOW else \
            self.OFFSET_WINDOW_SIZE if self.WINDOW_SIZE > self.OFFSET_WINDOW_SIZE else self.WINDOW_SIZE
        self.offset_window_beg = (time - self.offset_ws_actual + 1) if (time > (2 * self.B - 1 + self.offset_ws_actual)) \
            else (2 * self.B - 1)
        # calculating averages
        if self.offset_window_beg == 2 * self.B - 1:
            self.diff_avg_1.insert(time, np.NaN)
            self.diff_avg_2.insert(time, np.NaN)
            self.values_mean.insert(time, np.NaN)
            self.pred_mean_1.insert(time, np.NaN)
            self.pred_mean_2.insert(time, np.NaN)
            self.offset_retrain_trigger_1.insert(time, np.NaN)
            self.offset_retrain_trigger_2.insert(time, np.NaN)
            self.offset_retrain_signal_1.insert(time, np.NaN)
            self.offset_retrain_signal_2.insert(time, np.NaN)
            self.offset_retrain_threshold.insert(time, np.NaN)
        else:
            self.values_mean.insert(time, np.mean(self.values[self.offset_window_beg:time + 1]))
            self.pred_mean_1.insert(time, np.mean(self.predicted_1[self.offset_window_beg:time + 1]))
            self.pred_mean_2.insert(time, np.mean(self.predicted_2[self.offset_window_beg:time + 1]))
            self.diff_avg_1.insert(time, abs(self.values_mean[time] - self.pred_mean_1[time]))
            self.diff_avg_2.insert(time, abs(self.values_mean[time] - self.pred_mean_2[time]))

        # decide when to trigger LSTM retrain
        if time > 2 * self.B - 1 + self.offset_ws_actual:
            above_counter_1 = 0
            above_counter_2 = 0
            self.offset_retrain_threshold.insert(time, np.std(self.values[self.offset_window_beg:time + 1]))
            for y in range(self.offset_window_beg, time + 1):
                if self.diff_avg_1[y] > self.offset_retrain_threshold[y]:
                    above_counter_1 += 1
                if self.diff_avg_2[y] > self.offset_retrain_threshold[y]:
                    above_counter_2 += 1
            if above_counter_1 > int(self.offset_ws_actual * self.offset_percentage_curr_1):
                self.offset_retrain_signal_1.insert(time, True)
            else:
                self.offset_retrain_signal_1.insert(time, False)
            if above_counter_2 > int(self.offset_ws_actual * self.offset_percentage_curr_2):
                self.offset_retrain_signal_2.insert(time, True)
            else:
                self.offset_retrain_signal_2.insert(time, False)

            # actually trigger LSTM retrain if necessary for detector 1
            if self.offset_retrain_signal_1[self.offset_window_beg] and \
                    check_signal_in_ow(self.offset_window_beg, time, self.pattern_change_1) and \
                    check_signal_in_ow(self.offset_window_beg, time, self.anomaly_1) and \
                    self.offset_retrain_signal_1[time]:
                trigger_lstm_retrain(self, time, self.lstm_model_1, self.tmp_lstm_model_1, self.predicted_1)
                self.offset_retrain_trigger_1.insert(time, True)
            else:
                self.offset_retrain_trigger_1.insert(time, False)

            # actually trigger LSTM retrain if necessary for detector 2
            if self.offset_retrain_signal_2[self.offset_window_beg] and \
                    check_signal_in_ow(self.offset_window_beg, time, self.pattern_change_2) and \
                    check_signal_in_ow(self.offset_window_beg, time, self.anomaly_2) and \
                    self.offset_retrain_signal_2[time]:
                trigger_lstm_retrain(self, time, self.lstm_model_2, self.tmp_lstm_model_2, self.predicted_2)
                self.offset_retrain_trigger_2.insert(time, True)
            else:
                self.offset_retrain_trigger_2.insert(time, False)


def trigger_lstm_retrain(self, time, lstm_model, tmp_lstm_model, predicted):
    # retrain lstm model with the last B data points
    tmp_lstm_model.train_lstm(self.values[time - self.B:time], self.NUM_EPOCHS, self.DEBUG)
    # predict the next value again using the new model
    if time < self.length - 1:
        predicted[time] = tmp_lstm_model.predict_lstm(self.values[time - self.B + 1:time], time, self.DEBUG,
                                                      self.length, self.values[time + 1])
    # replace old LSTM model with the new one
    lstm_model.set_weights(tmp_lstm_model)
    # predict next value using the new model
    if time < self.length - 1:
        predicted.insert(time + 1, lstm_model.predict_lstm(self.values[time - self.B + 2:time + 1], time,
                                                           self.DEBUG, self.length, self.values[time + 1]))
