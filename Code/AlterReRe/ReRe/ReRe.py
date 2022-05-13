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
    # original parameters
    B = 30  # look-back param.
    F = 1  # predict-forward param. (NOT USED)
    THRESHOLD_STRENGTH = 3  # this is the coefficient of sigma (std. dev.), def.=3

    # Alter-ReRe parameters
    USE_WINDOW = True  # whether to use windowed-mode execution
    WINDOW_SIZE = 1000  # the size of the window when calculating AARE and thd values
    USE_AGING = True  # whether to turn on aging of values when calculating AARE and thd
    USE_AARE_AGING = True
    USE_THD_AGING = False
    AGE_POWER = 2

    # LSTM parameters
    NUM_EPOCHS = 30  # number of epochs
    NUM_NEURONS = 30  # number of neurons

    # implementation parameters
    FILENAME = 'ec2_cpu_utilization_ac20cd.csv'  # the name of the imported file
    TO_CSV = True  # whether to dump result data to a .csv file
    DO_DIFF = False  # whether to difference the data
    DO_SCAL = False  # whether to scale data between [-1,1]
    DO_DIV = True  # whether to divide data by the largest number in the dataset times 1,1
    USE_LESS = False  # whether to use only the beginning of the data
    LESS = 100  # the number of data points to use
    DEBUG = False  # whether to print verbose information to the console while operating
    NOTES = 'notes'  # type notes here to be saved with the hyperparameters
    STATUS_BAR = True  # replaces the #/# lines showing the algorithm operation with a status bar if DEBUG == False
    BATCH_STATUS_BAR = False

    ###############################
    # END OF USER SET PARAMETERS! #
    ###############################

    run_for = 1
    data = pd.DataFrame()
    length = 0

    # reading parameters from the specified file
    if OPERATION == 'file':
        runs_data = pd.read_csv(TESTRUN_FOLDER + '/' + TESTRUN_FILE)
        b_s = runs_data['B']
        thr_s = runs_data['THRESHOLD_STRENGTH']
        usw_s = runs_data['USE_WINDOW']
        wis_s = runs_data['WINDOW_SIZE']
        usa_s = runs_data['USE_AGING']
        uaa_s = runs_data['USE_AARE_AGING']
        uta_s = runs_data['USE_THD_AGING']
        agp_s = runs_data['AGE_POWER']
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
    from .preprocess import preprocess
    from .to_csv import init_to_csv, dump_hyperparameters, dump_results, write_time
    from .window_ageing import update_window_beginning, ageing_coefficient
    from .thd_aare_func import aare, thd_1, thd_2
