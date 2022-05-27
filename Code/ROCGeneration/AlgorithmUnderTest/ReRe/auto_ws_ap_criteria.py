"""
CRITERIA FUNCTIONS FOR EXTREME VALUES OF WS AND AP

These functions detect signs of too high or too low values of the WINDOW_SIZE and AGE_POWER parameters.
"""


def check_anom_flap(self, time, anom, patt):
    # calculating database
    if time - self.SIGNAL_DATABASE_LEN + 1 <= self.B:
        database_beginning = self.B
    else:
        database_beginning = time - self.SIGNAL_DATABASE_LEN + 1

    # detect flapping
    for y in range(database_beginning + 1, time + 1):
        anom_length = 0
        no_length = 0
        # if the start of an anomaly is detected, ...
        if (not anom[y - 1]) and anom[y]:
            # ... start counting its length
            for z in range(y, time + 1):
                # if the anomaly is over, stop the count
                if anom[z - 1] and (not anom[z]):
                    break
                else:
                    anom_length += 1
            # we have counted the length of the anomaly, then let's count the no's after it
            for zz in range(y + anom_length, time + 1):
                # if we have reached the anomaly length, or found a pattern change, break, there is no flapping
                if no_length > self.FLAPPING_LENGTH_COEFF * anom_length or patt[zz]:
                    break
                elif (not anom[zz - 1]) and anom[zz]:
                    if self.DEBUG:
                        print('\t\t\tanomaly flapping: True')
                    return True
                else:
                    no_length += 1

    if self.DEBUG:
        print('\t\t\tanomaly flapping: False')
    return False


def check_freq_sig(self, time, anom, patt, orig):
    # calculating database
    if time - self.SIGNAL_DATABASE_LEN + 1 <= self.B:
        database_beginning = self.B
    else:
        database_beginning = time - self.SIGNAL_DATABASE_LEN + 1

    # calculate the signal threshold based on the original data in the window
    sum_ = 0
    for y in range(self.window_beginning, time):
        # sum_ += abs(orig[y + 1] - orig[y])
        sum_ += (orig[y + 1] - orig[y]) ** 2
    signal_threshold = self.SIGNAL_THRESHOLD_COEFF * sum_ / (time - self.window_beginning)

    # calculate the signal ratio based on the number of signals in the signal database
    signalcount = 0
    for y in range(database_beginning + 1, time + 1):
        if ((not anom[y - 1]) and anom[y]) or (patt[y] and not patt[y - 1]):
            signalcount += 1
    signal_ratio_dat = signalcount / (time - database_beginning)

    if self.DEBUG:
        print('\t\t\tfrequent signalling: {}'.format(str(signal_ratio_dat > signal_threshold)))
    return signal_ratio_dat > signal_threshold


def check_long_anom(self, time, anom):
    # calculating database
    if time - self.SIGNAL_DATABASE_LEN + 1 <= self.B:
        database_beginning = self.B
    else:
        database_beginning = time - self.SIGNAL_DATABASE_LEN + 1

    # detect long anomalies (longer than TOO_LONG_ANOM_COEFF*B)
    for y in range(database_beginning + 1, time + 1):
        anom_length = 0
        # if the start of an anomaly is detected, ...
        if (not anom[y - 1]) and anom[y]:
            # ... start counting its length
            for z in range(y, time + 1):
                # if the anomaly is over, stop the count
                if anom[z - 1] and (not anom[z]):
                    break
                else:
                    anom_length += 1
            # if the counted length is larger than TOO_LONG_ANOM_COEFF*B, signal a long anomaly
            if anom_length > self.TOO_LONG_ANOM_COEFF * self.B:
                if self.DEBUG:
                    print('\t\t\tlong anomalies: True')
                return True

    if self.DEBUG:
        print('\t\t\tlong anomalies: False')
    return False


def check_no_sig(self, time_, anom_, patt_):
    # I've decided not to include this check in the algorithm, as it assumes a minimal number of detections
    # and if there are no anomalies in the dataset, and the pattern is not changing, no signals are a
    # perfectly valid symptom, therefore:
    if self.DEBUG:
        print('\t\t\tno signalling: disabled (False)')
    return False


def check_small_area(self, ws, ap):
    area_min = (self.B - 1) / 2
    area = (ws - 1) / (ap + 1)
    return area < area_min


def check_large_area(self, ws, ap):
    area_max = self.B ** 2
    area = (ws - 1) / (ap + 1)
    return area > area_max
