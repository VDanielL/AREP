"""
MAIN ALGORITHM OF RERE

This file contains the initialization and operation of the ReRe algorithm.
"""

import numpy as np
import ReRe.status_bar as bar


def initialize_rere(self):
    self.values = self.data['value'].to_numpy()

    # the first B-1 indices won't be used for predicted and AARE (first used at t=B)
    for i in range(0, self.B):
        self.predicted_1.insert(i, np.NaN)
        self.predicted_2.insert(i, np.NaN)
        self.AARE_1.insert(i, np.NaN)
        self.AARE_2.insert(i, np.NaN)
    # the first 2B-2 indices won't be used for threshold, anomaly and pattern_change (first used at t = 2B - 1)
    for i in range(0, 2 * self.B - 1):
        self.threshold_1.insert(i, np.NaN)
        self.threshold_2.insert(i, np.NaN)
        self.anomaly_1.insert(i, np.NaN)
        self.anomaly_2.insert(i, np.NaN)
        self.anomaly_aggr.insert(i, np.NaN)
        self.pattern_change_1.insert(i, np.NaN)
        self.pattern_change_2.insert(i, np.NaN)

    if self.OPERATION == 'file':
        print('\n' + '/' + '-' * 30)
        print('| TESTRUN {} OF {}'.format(self.inst_num + 1, self.run_for))
        print('\\' + '-' * 30 + '\n')


def next_timestep(self, time):
    if self.STATUS_BAR and not self.DEBUG:
        if time > self.B - 1:
            if self.BATCH_STATUS_BAR:
                with type(self).class_lock:
                    self.progress_for_executions[self.inst_num] = time
                    bar.draw_batch_status_bar(self.progress_for_executions, self.executions_max, self.run_for)
            else:
                bar.draw_status_bar(time, self.length)
    else:
        to_print = 'ReRe\'s begun timestep {}/{}' if self.DEBUG else '{}/{}'
        print(to_print.format(time, self.length - 1))

    # step 1: collecting a few (B) data points
    if time < self.B - 1:
        if self.DEBUG:
            print('\tStep 1')

    # step 2: train model based on the first B data points
    elif time == self.B - 1:
        if self.DEBUG:
            print('\tStep 2')
        # training the model
        self.lstm_model_1.train_lstm(self.values[0:self.B], self.NUM_EPOCHS, self.DEBUG)
        self.lstm_model_2.set_weights(self.lstm_model_1)
        # predicting using the model
        self.predicted_1.insert(time + 1, self.lstm_model_1.predict_lstm(self.values[1:self.B], time, self.DEBUG,
                                                                         self.length,
                                                                         self.values[time + 1]))
        self.predicted_2 = self.predicted_1.copy()

    # step 3: preparing for detection by collecting AAREs
    elif self.B - 1 < time < 2 * self.B - 1:
        if self.DEBUG:
            print('\tStep 3')
        # adding the new AARE value to the list
        self.AARE_1.insert(time, self.aare(time, self.predicted_1))
        self.AARE_2 = self.AARE_1.copy()
        # training the model
        self.lstm_model_1.train_lstm(self.values[time - self.B + 1:time + 1], self.NUM_EPOCHS, self.DEBUG)
        self.lstm_model_2.set_weights(self.lstm_model_1)
        # predicting using the model
        self.predicted_1.insert(time + 1, self.lstm_model_1.predict_lstm(self.values[time - self.B + 2:time + 1], time,
                                                                         self.DEBUG, self.length,
                                                                         self.values[time + 1]))
        self.predicted_2 = self.predicted_1.copy()

    # step 4: anomaly detection with ReRe and the two detectors
    elif time >= 2 * self.B - 1:
        if self.DEBUG:
            print('\tStep 4')

        # DETECTOR 1
        # adding the new AARE value to the list
        self.AARE_1.insert(time, self.aare(time, self.predicted_1))
        # calculating the threshold value
        self.threshold_1.insert(time, self.thd_1(time))
        # decide if there is an anomaly
        if self.AARE_1[time] <= self.threshold_1[time]:
            if self.DEBUG:
                print('\t\tNO_1 anomaly found.')
            # it ISN'T an anomaly
            detection_is_not_anomaly(self, time, self.anomaly_1, self.pattern_change_1, self.predicted_1,
                                     self.lstm_model_1)
        else:
            # it MIGHT BE an anomaly - further investigation needed
            # retrain lstm model with the last B data points
            self.tmp_lstm_model_1.train_lstm(self.values[time - self.B:time],
                                             self.NUM_EPOCHS, self.DEBUG)
            # predict the next value again using the new model
            if time < self.length - 1:
                self.predicted_1[time] = self.tmp_lstm_model_1.predict_lstm(self.values[time - self.B + 1:time], time,
                                                                            self.DEBUG, self.length,
                                                                            self.values[time + 1])
            # recalculate current AARE
            self.AARE_1[time] = self.aare(time, self.predicted_1)
            if self.AARE_1[time] <= self.threshold_1[time]:
                if self.DEBUG:
                    print('\t\tPATTERN CHANGE_1 found.')
                # it ISN'T an anomaly, only the patterns are changing
                detection_is_not_anomaly_but_pattern_change(self, time, self.anomaly_1, self.pattern_change_1,
                                                            self.predicted_1, self.lstm_model_1, self.tmp_lstm_model_1)
            else:
                # it IS an anomaly
                detection_is_anomaly(self, time, self.anomaly_1, self.pattern_change_1, self.predicted_1,
                                     self.lstm_model_1, 'ANOMALY_1')

        # DETECTOR 2
        # adding the new AARE value to the list
        self.AARE_2.insert(time, self.aare(time, self.predicted_2))
        # calculating the threshold value
        self.threshold_2.insert(time, self.thd_2(time))
        # decide if there is an anomaly
        if self.AARE_2[time] <= self.threshold_2[time]:
            if self.DEBUG:
                print('\t\tNO_2 anomaly found.')
            # it ISN'T an anomaly
            detection_is_not_anomaly(self, time, self.anomaly_2, self.pattern_change_2, self.predicted_2,
                                     self.lstm_model_2)
        else:
            # it MIGHT BE an anomaly - further investigation needed
            # retrain lstm model with the last B data points
            self.tmp_lstm_model_2.train_lstm(self.values[time - self.B:time],
                                             self.NUM_EPOCHS, self.DEBUG)
            # predict the next value again using the new model
            if time < self.length - 1:
                self.predicted_2[time] = self.tmp_lstm_model_2.predict_lstm(self.values[time - self.B + 1:time], time,
                                                                            self.DEBUG, self.length,
                                                                            self.values[time + 1])
            # recalculate current AARE
            self.AARE_2[time] = self.aare(time, self.predicted_2)
            if self.AARE_2[time] <= self.threshold_2[time]:
                if self.DEBUG:
                    print('\t\tPATTERN CHANGE_2 found.')
                # it ISN'T an anomaly, only the patterns are changing
                detection_is_not_anomaly_but_pattern_change(self, time, self.anomaly_2, self.pattern_change_2,
                                                            self.predicted_2, self.lstm_model_2, self.tmp_lstm_model_2)
            else:
                # it IS an anomaly
                detection_is_anomaly(self, time, self.anomaly_2, self.pattern_change_2, self.predicted_2,
                                     self.lstm_model_2, 'ANOMALY_2')

        # deciding if both detectors show an anomaly
        if self.anomaly_1[time] and self.anomaly_2[time]:
            self.anomaly_aggr.insert(time, True)
            if not self.STATUS_BAR:
                print('ANOMALY at timestep {}!'.format(time))
        else:
            self.anomaly_aggr.insert(time, False)
            if self.DEBUG:
                print('NO anomaly at timestep {}!'.format(time))


def detection_is_not_anomaly(rere, time, anomaly_list, pattern_change_list, predicted_list, lstm_model):
    anomaly_list.insert(time, False)
    pattern_change_list.insert(time, False)
    if time < len(rere.values) - 1:
        predicted_list.insert(time + 1,
                              lstm_model.predict_lstm(rere.values[time - rere.B + 2:time + 1],
                                                      time, rere.DEBUG, rere.length,
                                                      rere.values[time + 1]))


def detection_is_not_anomaly_but_pattern_change(rere, time, anomaly_list, pattern_change_list, predicted_list,
                                                lstm_model, tmp_lstm_model):
    anomaly_list.insert(time, False)
    pattern_change_list.insert(time, True)
    # replace old LSTM model with the new one
    lstm_model.set_weights(tmp_lstm_model)
    # predict next value using the new model
    if time < len(rere.values) - 1:
        predicted_list.insert(time + 1,
                              lstm_model.predict_lstm(rere.values[time - rere.B + 2:time + 1],
                                                      time, rere.DEBUG, rere.length,
                                                      rere.values[time + 1]))


def detection_is_anomaly(rere, time, anomaly_list, pattern_change_list, predicted_list, lstm_model, anomaly_number):
    anomaly_list.insert(time, True)
    pattern_change_list.insert(time, False)
    if not rere.STATUS_BAR:
        print('{0} at timestep {1}!'.format(anomaly_number, time))
    # predict the next value based on the old model
    if time < len(rere.values) - 1:
        predicted_list.insert(time + 1,
                              lstm_model.predict_lstm(rere.values[time - rere.B + 2:time + 1],
                                                      time, rere.DEBUG, rere.length,
                                                      rere.values[time + 1]))
