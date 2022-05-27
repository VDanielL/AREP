"""
AUTOMATIC OFFSET COMPENSATION PARAMETER TUNING

This file contains the algorithm for automatically tuning offset compensation parameter OP.
"""

import numpy as np


def init_auto_offset_compensation(self):
    if self.USE_OFFSET_COMP and self.USE_AUTOMATIC_OFFSET:
        self.offset_percentage_1 = list()
        self.offset_percentage_2 = list()
        self.retrain_percentage_1 = list()
        self.retrain_percentage_2 = list()

        # the first 2B-2 indices won't be used for these (first used at t = 2B - 1)
        for i in range(0, 2 * self.B - 1):
            self.offset_percentage_1.insert(i, np.NaN)
            self.offset_percentage_2.insert(i, np.NaN)
            self.retrain_percentage_1.insert(i, np.NaN)
            self.retrain_percentage_2.insert(i, np.NaN)


def auto_tune_offset(self, time):
    # automatic tuning of parameters, if allowed:
    if self.USE_AUTOMATIC_OFFSET:
        if time > 2 * self.B - 1 + 2 * self.offset_ws_actual:
            # count the number of retrains in the offset window
            rp_tmp_1 = 0
            rp_tmp_2 = 0
            for y in range(self.offset_window_beg, time + 1):
                if self.pattern_change_1[y] or self.anomaly_1[y] or self.offset_retrain_trigger_1[y]:
                    rp_tmp_1 += 1
                if self.pattern_change_2[y] or self.anomaly_2[y] or self.offset_retrain_trigger_2[y]:
                    rp_tmp_2 += 1
            # calculate RP for both detectors
            self.retrain_percentage_1.insert(time, rp_tmp_1 / self.offset_ws_actual)
            self.retrain_percentage_2.insert(time, rp_tmp_2 / self.offset_ws_actual)

            # calculate RP_MAX
            # for ReRe, we divide ACCEPTABLE_AVG_DURATION by two as both detectors compete for resources
            RETRAIN_PERCENTAGE_LIMIT = ((self.offset_ws_actual - 1) / self.offset_ws_actual) * \
                                       (((self.ACCEPTABLE_AVG_DURATION / 2) - self.avg_dur_normal) /
                                        (self.avg_dur_lstm - self.avg_dur_normal))

            # if RP is above a limit, raise OP to the maximum for detector 1
            if self.retrain_percentage_1[time] > RETRAIN_PERCENTAGE_LIMIT:
                self.offset_percentage_curr_1 = 1
            # otherwise reduce it gradually over the offset window
            else:
                if self.offset_percentage_curr_1 > 1 / self.offset_ws_actual:
                    self.offset_percentage_curr_1 -= 1 / self.offset_ws_actual
                else:
                    self.offset_percentage_curr = 0

            # if RP is above a limit, raise OP to the maximum for detector 2
            if self.retrain_percentage_2[time] > RETRAIN_PERCENTAGE_LIMIT:
                self.offset_percentage_curr_2 = 1
            # otherwise reduce it gradually over the offset window
            else:
                if self.offset_percentage_curr_2 > 1 / self.offset_ws_actual:
                    self.offset_percentage_curr_2 -= 1 / self.offset_ws_actual
                else:
                    self.offset_percentage_curr = 0

            # save settings
            self.offset_percentage_1.insert(time, self.offset_percentage_curr_1)
            self.offset_percentage_2.insert(time, self.offset_percentage_curr_2)

        else:
            self.retrain_percentage_1.insert(time, np.NaN)
            self.retrain_percentage_2.insert(time, np.NaN)
            self.offset_percentage_1.insert(time, np.NaN)
            self.offset_percentage_2.insert(time, np.NaN)
