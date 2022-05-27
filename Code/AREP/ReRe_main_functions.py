from matplotlib import pyplot

from ReRe import ReRe
import time as timelib


def run_rere(test_instance, progress_for_executions, executions_max):
    # create new instance of ReRe
    rere = ReRe.ReRe()
    rere.init_timer()
    rere.init_offset_compensation()
    rere.init_auto_offset_compensation()
    rere.init_auto_ws_ap()

    # setting parameters automatically from the specified file
    if rere.OPERATION == 'file':
        rere.param_refresh(test_instance)
        rere.inst_num = test_instance
        rere.progress_for_executions = progress_for_executions
        rere.BATCH_STATUS_BAR = True

    # load data file into the data frame and initialize operation parameters
    rere.load()
    if rere.OPERATION == 'file':
        rere.executions_max = executions_max
        executions_max[rere.inst_num] = rere.length
    rere.initialize_cons()

    # comparing original and preprocessed data (1)
    if rere.OPERATION == 'manual':
        fig, axs = pyplot.subplots(2, sharex='all')
        axs[0].plot_date(rere.data.timestamp, rere.data.value, marker='', linestyle='-', label='original values')
        axs[0].set_title('Original data')
        axs[0].legend()
        axs[0].grid()

    # preprocess the dataset
    rere.preprocess()

    # comparing original and preprocessed data (2)
    if rere.OPERATION == 'manual':
        axs[1].plot_date(rere.data.timestamp, rere.data.value, marker='', linestyle='-', label='preprocessed data')
        axs[1].set_title('Preprocessed data')
        fig.autofmt_xdate()
        axs[1].legend()
        axs[1].grid()
        pyplot.show()

    # ReRe algorithm starts now, starting timer
    timer_start = timelib.time()

    # initialize ReRe algorithm
    rere.initialize_rere()

    # dump parameters to results_yyyymmddhhmmss.csv
    if rere.TO_CSV:
        data_names, param_filename, result_filename = rere.init_to_csv()
        rere.dump_hyperparameters(timer_start, param_filename)

    # THE ACTUAL RERE ALGORITHM
    time = 0  # current timestep parameter
    while time < rere.length:
        # start time measurement
        rere.start_timestep()

        # update the beginning of the sliding window
        rere.update_window_beginning(time)

        # perform one timestep of the original ReRe algorithm
        rere.next_timestep(time)

        # perform offset compensation
        if rere.USE_OFFSET_COMP:
            rere.compensate_offset(time)

        # perform automatic tuning of offset compensation
        if rere.USE_AUTOMATIC_OFFSET:
            rere.auto_tune_offset(time)

        # perform automatic tuning of WINDOW_SIZE and AGE_POWER
        if rere.USE_AUTOMATIC_WS_AP:
            rere.auto_tune_ws_ap(time)

        # stop time measurement and update averages
        rere.end_timestep(time)

        # dump results to a .csv file
        if rere.TO_CSV:
            rere.dump_results(time, data_names, result_filename)
            rere.write_time(timer_start, param_filename)

        # jump to the next timestep
        time += 1

    # end
    print('\nReRe over, total time: {} seconds.'.format(round(timelib.time() - timer_start, 2)))
    time_data = '\nOf this, there were {} LSTM retrains, taking {} s on average (both measured in Detector 1).\n' + \
                'Normal timesteps took {} s on average'
    print(time_data.format(rere.retrain_count, round(rere.avg_dur_lstm, 2), round(rere.avg_dur_normal, 2)))

    if rere.OPERATION == 'file':
        print('Finished automatic test run based on {}. {} of {}.'.format(rere.TESTRUN_FILE, test_instance + 1,
                                                                          rere.run_for + 1))
