"""
TIMING ALGORITHM AND FUNCTIONS

This file contains the necessary tools for keeping track of the duration of various types of self timesteps.
WE ONLY MEASURE DETECTOR 1 AND INFER DATA FOR DETECTOR 2 FROM THAT.
"""

import time as timelib


def init_timer(self):
    self.timesteps_dur = list()
    self.avg_dur_normal = 0.0
    self.avg_dur_lstm = 0.0
    self.retrain_count = 0
    self.timestep_begin = 0.0
    self.current_duration = 0.0


def start_timestep(self):
    self.timestep_begin = timelib.time()


def end_timestep(self, time):
    # calculating and updating average timestep durations
    # count the duration of the current timestep
    self.current_duration = timelib.time() - self.timestep_begin
    self.timesteps_dur.insert(time, self.current_duration)

    # initial value for normal timesteps
    if time == 0:
        self.avg_dur_normal = self.current_duration
    # fixed 1normal timesteps
    elif 0 < time < self.B - 1:
        self.avg_dur_normal = (self.avg_dur_normal * (time - self.retrain_count) + self.current_duration) * \
                              (1 / (time - self.current_duration + 1))
    # initial value for LSTM timesteps
    elif time == self.B - 1:
        self.avg_dur_lstm = self.current_duration
        self.retrain_count = 1
    # fixed 1LSTM timesteps
    elif self.B - 1 < time < 2 * self.B - 1:
        self.avg_dur_lstm = (self.retrain_count * self.avg_dur_lstm + self.current_duration) * (
                    1 / (self.retrain_count + 1))
        self.retrain_count += 1
    # using offset compensation, retrains can occur due to either a pattern change, an anomaly or an offset retrain
    elif self.USE_OFFSET_COMP:
        # LSTM_1 + LSTM_2 == 2 -> 2LSTM timestep
        if (self.pattern_change_1[time] or self.anomaly_1[time] or self.offset_retrain_trigger_1[time]) + \
                (self.pattern_change_2[time] or self.anomaly_2[time] or self.offset_retrain_trigger_2[time]) == 2:
            self.avg_dur_lstm = (self.retrain_count * self.avg_dur_lstm + (self.current_duration / 2)) * (
                        1 / (self.retrain_count + 1))
            self.retrain_count += 1
        # LSTM_1 + LSTM_2 == 1 -> 1LSTM + 1normal ~ 1LSTM timestep
        elif (self.pattern_change_1[time] or self.anomaly_1[time] or self.offset_retrain_trigger_1[time]) + \
                (self.pattern_change_2[time] or self.anomaly_2[time] or self.offset_retrain_trigger_2[time]) == 1:
            self.avg_dur_lstm = (self.retrain_count * self.avg_dur_lstm + self.current_duration) * (
                        1 / (self.retrain_count + 1))
            self.retrain_count += 1
        # LSTM_1 + LSTM_2 == 0 -> 2normal timestep
        else:
            self.avg_dur_normal = (self.avg_dur_normal * (time - self.retrain_count) + (self.current_duration / 2)) * \
                                  (1 / (time - self.current_duration + 1))
    # if we don't use offset compensation, retrains can occur only as a result of a pattern change or an anomaly
    else:
        # LSTM_1 + LSTM_2 == 2 -> 2LSTM timestep
        if (self.pattern_change_1[time] or self.anomaly_1[time]) + \
                (self.pattern_change_2[time] or self.anomaly_2[time]) == 2:
            self.avg_dur_lstm = (self.retrain_count * self.avg_dur_lstm + (self.current_duration / 2)) * (
                        1 / (self.retrain_count + 1))
            self.retrain_count += 2
        # LSTM_1 + LSTM_2 == 1 -> 1LSTM + 1normal ~ 1LSTM timestep
        elif (self.pattern_change_1[time] or self.anomaly_1[time]) + \
                (self.pattern_change_2[time] or self.anomaly_2[time]) == 1:
            self.avg_dur_lstm = (self.retrain_count * self.avg_dur_lstm + self.current_duration) * (
                        1 / (self.retrain_count + 1))
            self.retrain_count += 1
        # LSTM_1 + LSTM_2 == 0 -> 2normal timestep
        else:
            self.avg_dur_normal = (self.avg_dur_normal * (time - self.retrain_count) + (self.current_duration / 2)) * \
                                  (1 / (time - self.current_duration + 1))
