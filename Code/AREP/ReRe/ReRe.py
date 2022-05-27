"""
INITIALIZATION PARAMETERS OF RERE

These variables contain the hyperparameters for ReRe operation.
"""


import numpy as np
import pandas as pd
import multiprocessing


class ReRe:
    # class variables for multiprocessing
    class_lock = multiprocessing.Lock()
    progress_for_executions = {}
    executions_max = {}
    inst_num = 0

    #########################
    # SET THESE PARAMETERS! #
    #########################

    OPERATION = 'manual'
    # 'manual' - set parameters by hand,
    # 'file' - set parameters in a separate file (can be multiple tests)

    if OPERATION == 'file':
        TESTRUN_FOLDER = 'testRuns'  # set the folder of the test runs here
        TESTRUN_FILE = 'testrun.csv'  # set filename of the test run here

    # set 'manual' parameters here ('file' mode will overwrite these):
    # RePAD parameters
    B = 30  # look-back param.
    F = 1  # predict-forward param.
    THRESHOLD_STRENGTH = 3  # this is the coefficient of sigma (std. dev.), def.=3
    USE_WINDOW = True  # whether to use windowed-mode execution
    WINDOW_SIZE = 800  # the size of the window when calculating AARE and thd values

    USE_AGING = True  # whether to turn on aging of values when calculating AARE and thd
    USE_AARE_AGING = True
    USE_THD_AGING = False
    AGE_POWER = 2.5

    USE_AUTOMATIC_WS_AP = True

    USE_OFFSET_COMP = True  # whether to use the implemented offset correction segment that triggers a model retrain
    ACCEPTABLE_AVG_DURATION = 1  # [seconds]: the maximum allowed duration of one timestep on average
    USE_AUTOMATIC_OFFSET = True
    OFFSET_WINDOW_SIZE = 30  # the number of datapoints to use with offset compensation
    OFFSET_PERCENTAGE = .9  # the ratio of datapoints that need to be above threshold to trigger LSTM retrain

    # LSTM parameters
    NUM_EPOCHS = 30  # number of epochs
    NUM_NEURONS = 30  # number of neurons

    # implementation parameters
    FILENAME = 'ec2_cpu_utilization_ac20cd.csv'  # the name of the imported file
    TO_CSV = True  # whether to dump data to a .csv file
    EVAL_EXPORT = False  # whether to export data for evaluation use
    EVAL_FOLDER = 'eval_results'
    DO_DIV = True  # whether to divide data by the largest number in the dataset times 1,1
    USE_LESS = False  # whether to use only the beginning of the data
    LESS = 100  # the number of data points to use
    DEBUG = False  # whether to print verbose information to the console while operating
    NOTES = 'arep test'  # type notes here to be saved with the hyperparameters
    STATUS_BAR = True  # replaces the #/# lines showing the algorithm operation with a status bar if DEBUG == False
    BATCH_STATUS_BAR = False

    ###############################0,
    # END OF USER SET PARAMETERS! #
    ###############################

    run_for = 1
    data = pd.DataFrame()
    length = 0

    # reading parameters from the specified file
    if OPERATION == 'file':
        runs_data = pd.read_csv('../' + TESTRUN_FOLDER + '/' + TESTRUN_FILE)
        b_s = runs_data['B']
        thr_s = runs_data['THRESHOLD_STRENGTH']
        usw_s = runs_data['USE_WINDOW']
        wis_s = runs_data['WINDOW_SIZE']
        usa_s = runs_data['USE_AGING']
        uaa_s = runs_data['USE_AARE_AGING']
        uta_s = runs_data['USE_THD_AGING']
        agp_s = runs_data['AGE_POWER']
        uap_s = runs_data['USE_AUTOMATIC_WS_AP']
        uoc_s = runs_data['USE_OFFSET_COMP']
        aad_s = runs_data['ACCEPTABLE_AVG_DURATION']
        uao_s = runs_data['USE_AUTOMATIC_OFFSET']
        owi_s = runs_data['OFFSET_WINDOW_SIZE']
        ope_s = runs_data['OFFSET_PERCENTAGE']
        nep_s = runs_data['NUM_EPOCHS']
        nne_s = runs_data['NUM_NEURONS']
        fil_s = runs_data['FILENAME']

        run_for = len(b_s)

        TO_CSV = True
        DO_DIFF = False
        DO_SCAL = False
        DO_DIV = True
        USE_LESS = False
        DEBUG = False

    # initialize parameters for automatic window and ageing
    if USE_AUTOMATIC_WS_AP:
        WS_AP_COEFF = 2  # the number to multiply the WS and AP parameters by
        SIGNAL_DATABASE_LEN = B ** 2  # signal window size
        DECISION_FREQ = B  # the minimum number of timesteps between two tuning events
        FLAPPING_LENGTH_COEFF = 1.5  # the maximum number of times no_length can be larger than anom_length
        SIGNAL_THRESHOLD_COEFF = B  # the coefficient when calculating percentage threshold of 'freq_retrain'
        TOO_LONG_ANOM_COEFF = 2.5  # this times B is the maximum allowed length of an anomaly

    def __init__(self):
        from ReRe.lstm_func import Lstm

        # create LSTM models
        self.lstm_model_1 = Lstm(self.B, self.NUM_NEURONS)
        self.lstm_model_2 = Lstm(self.B, self.NUM_NEURONS)
        self.tmp_lstm_model_1 = Lstm(self.B, self.NUM_NEURONS)
        self.tmp_lstm_model_2 = Lstm(self.B, self.NUM_NEURONS)

        # create lists
        self.values = np.empty(self.length)
        self.predicted_1 = list()
        self.predicted_2 = list()
        self.AARE_1 = list()
        self.AARE_2 = list()
        self.threshold_1 = list()
        self.threshold_2 = list()
        self.anomaly_1 = list()
        self.anomaly_2 = list()
        self.anomaly_aggr = list()
        self.pattern_change_1 = list()
        self.pattern_change_2 = list()

        # aging
        self.window_beginning = 0

    from .initial import param_refresh, load, initialize_cons
    from .main_algo import initialize_rere, next_timestep
    from .offset_comp import init_offset_compensation, compensate_offset
    from .auto_offset_comp import init_auto_offset_compensation, auto_tune_offset
    from .timer import init_timer, start_timestep, end_timestep
    from .preprocess import preprocess
    from .auto_ws_ap import init_auto_ws_ap, auto_tune_ws_ap
    from .to_csv import init_to_csv, dump_hyperparameters, dump_results, write_time
    from .window_ageing import update_window_beginning, ageing_coefficient
    from .thd_aare_func import aare, thd_1, thd_2
