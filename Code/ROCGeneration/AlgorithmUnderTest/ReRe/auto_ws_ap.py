"""
AUTOMATIC TUNING OF WS AND AP PARAMETERS

This file contains the algorithm for automatically tuning the size of the sliding window and the age power parameter.
These parameters are tuned based on the output of Detector 1.
"""

import AlgorithmUnderTest.ReRe.auto_ws_ap_criteria as criteria
import numpy as np


def init_auto_ws_ap(self):
    if self.USE_AUTOMATIC_WS_AP:
        self.window_size = list()
        self.age_power = list()
        self.anom_flap = list()
        self.freq_sig = list()
        self.long_anom = list()
        self.no_sig = list()
        self.window_size.insert(0, self.WINDOW_SIZE)
        self.age_power.insert(0, self.AGE_POWER)
        self.last_correction = 0
        self.last_low_WS = 0
        self.last_high_WS = np.inf
        self.last_low_AP = 0
        self.last_high_AP = np.inf
        self.op = 0  # tracks which state the tuning system is in (0 - none, 1 - WS done, 2 - AP done)


def auto_tune_ws_ap(self, time):
   # automatic WINDOW_SIZE and AGE_POWER component
    if self.USE_AUTOMATIC_WS_AP:
        # if the last adjustment happened more then B timesteps ago (and we are past the initial timesteps)
        if (time >= self.last_correction + self.DECISION_FREQ) and (time >= (2 * self.B - 1) + self.B):
            # adjusting WS
            if self.op == 0:
                self.op = 1
                if self.DEBUG:
                    print('\t\tAutomatic WS & AP/Step WS. Checking conditions...')

                # WS_MIN = anomaly flapping or frequent signals
                ws_min_1 = criteria.check_anom_flap(self, time, self.anomaly_1, self.pattern_change_1)
                self.anom_flap.insert(time, ws_min_1)
                ws_min_2 = criteria.check_freq_sig(self, time, self.anomaly_1, self.pattern_change_1, self.values)
                self.freq_sig.insert(time, ws_min_2)
                ws_min_3 = criteria.check_small_area(self, self.window_size[time-1], self.age_power[time-1])
                WS_MIN = ws_min_1 or ws_min_2 or ws_min_3

                # WS_MAX = long anomalies or no signals
                ws_max_1 = criteria.check_long_anom(self, time, self.anomaly_1)
                self.long_anom.insert(time, ws_max_1)
                ws_max_2 = criteria.check_no_sig(self, time, self.anomaly_1, self.pattern_change_1)
                self.no_sig.insert(time, ws_max_2)
                ws_max_3 = criteria.check_large_area(self, self.window_size[time - 1], self.age_power[time - 1])
                WS_MAX = ws_max_1 or ws_max_2 or ws_max_3

                # if WS_MIN and not WS_MAX -> increase WS
                # llW   WS * lhW
                if WS_MIN and (not WS_MAX):
                    if np.isinf(self.last_high_WS):
                        self.last_low_WS = self.WINDOW_SIZE
                        self.WINDOW_SIZE = int(self.WS_AP_COEFF * self.WINDOW_SIZE)

                    elif self.last_high_WS <= self.WINDOW_SIZE:
                        self.last_low_WS = self.WINDOW_SIZE
                        self.WINDOW_SIZE = int(self.WS_AP_COEFF * self.WINDOW_SIZE)
                        self.last_high_WS = self.WINDOW_SIZE

                    else:
                        self.last_low_WS = self.WINDOW_SIZE
                        self.WINDOW_SIZE = int((self.last_high_WS + self.WINDOW_SIZE) / 2)

                    self.last_correction = time

                # if WS_MAX and not WS_MIN -> decrease WS
                # llW * WS   lhW
                elif WS_MAX and (not WS_MIN):
                    if self.last_low_WS >= self.WINDOW_SIZE:
                        self.last_high_WS = self.WINDOW_SIZE
                        self.WINDOW_SIZE = int(self.WINDOW_SIZE / self.WS_AP_COEFF)
                        self.last_low_WS = self.WINDOW_SIZE
                    else:
                        self.last_high_WS = self.WINDOW_SIZE
                        self.WINDOW_SIZE = int((self.last_low_WS + self.WINDOW_SIZE) / 2)

                    self.last_correction = time

            # adjust AP by a smaller percentage, leave WS intact
            elif self.op == 1:
                self.op = 2
                if self.DEBUG:
                    print('\t\tAutomatic WS & AP/Step AP. Checking conditions...')

                # AP_MAX = anomaly flapping or frequent signals
                ap_max_1 = criteria.check_anom_flap(self, time, self.anomaly_1, self.pattern_change_1)
                self.anom_flap.insert(time, ap_max_1)
                ap_max_2 = criteria.check_freq_sig(self, time, self.anomaly_1, self.pattern_change_1, self.values)
                self.freq_sig.insert(time, ap_max_2)
                ap_max_3 = criteria.check_small_area(self, self.window_size[time - 1], self.age_power[time - 1])
                AP_MAX = ap_max_1 or ap_max_2 or ap_max_3

                # AP_MIN = long anomalies or no signals or a smaller than one AP
                ap_min_1 = criteria.check_long_anom(self, time, self.anomaly_1)
                self.long_anom.insert(time, ap_min_1)
                ap_min_2 = criteria.check_no_sig(self, time, self.anomaly_1, self.pattern_change_1)
                self.no_sig.insert(time, ap_min_2)
                ap_min_3 = criteria.check_large_area(self, self.window_size[time - 1], self.age_power[time - 1])
                AP_MIN = ap_min_1 or ap_min_2 or ap_min_3

                # llA   AP * lhA
                if AP_MIN and (not AP_MAX):
                    if np.isinf(self.last_high_AP):
                        self.last_low_AP = self.AGE_POWER
                        self.AGE_POWER = self.WS_AP_COEFF * self.AGE_POWER

                    elif self.last_high_AP <= self.AGE_POWER:
                        self.last_low_AP = self.AGE_POWER
                        self.AGE_POWER = self.WS_AP_COEFF * self.AGE_POWER
                        self.last_high_AP = self.AGE_POWER

                    else:
                        self.last_low_AP = self.AGE_POWER
                        self.AGE_POWER = (self.last_high_AP + self.AGE_POWER) / 2

                    self.last_correction = time

                # llA * AP   lhA
                elif AP_MAX and (not AP_MIN):
                    if self.last_low_AP >= self.AGE_POWER:
                        self.last_high_AP = self.AGE_POWER
                        self.AGE_POWER = self.AGE_POWER / self.WS_AP_COEFF
                        self.last_low_AP = self.AGE_POWER
                    else:
                        self.last_high_AP = self.AGE_POWER
                        self.AGE_POWER = (self.last_low_AP + self.AGE_POWER) / 2

                    self.last_correction = time

            else:
                self.op = 0
                self.anom_flap.insert(time, np.NaN)
                self.freq_sig.insert(time, np.NaN)
                self.long_anom.insert(time, np.NaN)
                self.no_sig.insert(time, np.NaN)

        # else insert NAN to the databases of the four criteria
        else:
            self.anom_flap.insert(time, np.NaN)
            self.freq_sig.insert(time, np.NaN)
            self.long_anom.insert(time, np.NaN)
            self.no_sig.insert(time, np.NaN)

        # after tuning, insert parameters into the lists
        self.window_size.insert(time, self.WINDOW_SIZE)
        self.age_power.insert(time, self.AGE_POWER)

        if self.DEBUG:
            print('\t\t\tWINDOW_SIZE: {}, AGE_POWER: {}'.format(self.WINDOW_SIZE, self.AGE_POWER))
