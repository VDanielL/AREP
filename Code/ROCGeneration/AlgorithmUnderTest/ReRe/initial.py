"""
RERE INITIALIZATION FUNCTIONS

The param_refresh function replaces hyperparameter values with the next set from the file in 'file' mode.
The load function imports the selected dataset.
The initialize_cons function sets parameters to a self-consistent value.
"""

import pandas as pd


# set or update algorithm parameters from file
def param_refresh(self, test_instance):
    # setting parameters automatically from the specified file
    if self.OPERATION == 'file':
        self.B = int(self.b_s[test_instance])
        self.F = 1
        self.THRESHOLD_STRENGTH = float(self.thr_s[test_instance])
        self.USE_WINDOW = True if self.usw_s[test_instance] == 'T' else False
        if self.USE_WINDOW:
            self.WINDOW_SIZE = int(self.wis_s[test_instance])
        self.USE_AGING = True if self.usa_s[test_instance] == 'T' else False
        if self.USE_AGING:
            self.USE_AARE_AGING = True if self.uaa_s[test_instance] == 'T' else False
            self.USE_THD_AGING = True if self.uta_s[test_instance] == 'T' else False
            self.AGE_POWER = float(self.agp_s[test_instance])
        else:
            self.USE_AARE_AGING = False
            self.USE_THD_AGING = False
        if self.USE_AGING and self.USE_WINDOW:
            self.USE_AUTOMATIC_WS_AP = True if self.uap_s[test_instance] == 'T' else False
        else:
            self.USE_AUTOMATIC_WS_AP = False
        self.USE_OFFSET_COMP = True if self.uoc_s[test_instance] == 'T' else False
        if self.USE_OFFSET_COMP:
            self.ACCEPTABLE_AVG_DURATION = float(self.aad_s[test_instance])
            self.USE_AUTOMATIC_OFFSET = True if self.uao_s[test_instance] == 'T' else False
            self.OFFSET_WINDOW_SIZE = int(self.owi_s[test_instance])
            self.OFFSET_PERCENTAGE = float(self.ope_s[test_instance])
        else:
            self.USE_AUTOMATIC_OFFSET = False
        self.NUM_EPOCHS = int(self.nep_s[test_instance])
        self.NUM_NEURONS = int(self.nne_s[test_instance])
        self.FILENAME = self.fil_s[test_instance]
        self.NOTES = 'Automatic test run based on {}. {} of {}.'.format(self.TESTRUN_FILE, test_instance + 1,
                                                                        self.run_for + 1)


# importing & visualizing data
def load(self):
    with type(self).class_lock:
        self.data = pd.read_csv('datasets/' + self.FILENAME, parse_dates=['timestamp'])
        self.length = len(self.data)
        if self.DEBUG:
            print(self.data.info())
            print("\n")
            print(self.data.head())
            print("Missing values in the data:\n\n{}".format(self.data.isnull().sum()))


# set parameters to a self-consistent value
def initialize_cons(self):
    # setting other parameters to preserve consistency and initialize various parameters
    if not self.USE_WINDOW:
        self.WINDOW_SIZE = len(self.data)
    if not self.USE_AGING:
        self.USE_AARE_AGING = False
        self.USE_THD_AGING = False
        self.AGE_POWER = 0
    if (not self.USE_AGING) or (not self.USE_WINDOW):
        self.USE_AUTOMATIC_WS_AP = False
    if self.USE_OFFSET_COMP:
        if self.USE_AUTOMATIC_OFFSET:
            self.OFFSET_WINDOW_SIZE = 50
            self.OFFSET_PERCENTAGE = 0
    else:
        self.USE_AUTOMATIC_OFFSET = False

    if self.OPERATION == 'manual':
        self.EVAL_EXPORT = False
