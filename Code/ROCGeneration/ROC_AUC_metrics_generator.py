"""
This program draws the ROC curve using the algorithms internal decision logic.
This is done by adjusting the THRESHOLD_STRENGTH parameter when calculating the
threshold for the AARE error values.

Main structure of the algorithm is as follows.
Below algorithm is executed for all files in the specified folder, then results are compiled.

do for THRESHOLD_STEPS turns:
    adjust thd_st -> measure conf. matrix -> calculate metrics
save data
"""

import glob
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import multiprocessing
from copy import deepcopy


if __name__ == '__main__':
    import ROC_AUC_metrics_generator_functions as func
    thresholds = []

    ############
    # SETTINGS #
    ############

    DATA_FOLDER = 'forROC'  # folder to obtain the original datasets from
    FLAGS_FOLDER = 'flags'  # folder to obtain ground truth labels from
    METRICS_FOLDER = 'metrics'  # folder /ROC to save the data to
    READ_FOLDER = 'metrics/ROC'  # folder to read and plot the ROC from
    SCROLL_FOLDERS = ['metrics/ROC1',  # only used in 'scroll' mode to compare the results of more algorithms
                      'metrics/ROC2', 'metrics/ROC3']
    CHART_FOLDER = 'metrics'

    ROUND = 4
    DEBUG = False

    THRESHOLD_STEPS = 24
    THRESHOLD_LIMIT = 6
    THRESHOLD_CENTER = 2.75  # center the thresholds around this value (for corrected operation)
    THRESHOLD_INCREASE_MODE = 'fixed'
    # thd increase modes:
    # 'linear':     add a constant to the previous thd
    # 'corrected':  choose values around the center of the interval more densely, adjust for THRESHOLD_CENTER but
    #               keep min and max values
    # 'fixed':      given set of points

    ANOMALY_WINDOW_SIZE = 20
    ADJUST_SIGNALS = False  # whether to only take into account no->yes changes in anomaly signals
    NORMALIZE_ANOM_METRICS = True
    METRIC = 'classic'
    # metric settings:
    # 'classic':    use TPR and FPR metrics to draw the ROC curve and calculate the AUC
    # 'prec_rec':   use precision and recall to create the curve and calculate the AUC

    DRAW_THD_VALUES = False
    ROC_LOOK_MODE = 'lin_re'
    # making the ROC curve look nicer, modes:
    # 'raw':    draw original points, no adjustment
    # 'lin_re': linear interpolation by removing "bad" points that violate ROC monotonic increase
    # 'smooth': employ smoothing by preserving AUC metric, but also maintaining ROC monotonic increase

    OPERATION = 'draw'
    # operation settings:
    # 'save':   save ROC data of all results in the folder to a file
    # 'draw':   'save', then draw the ROC data of all results in the folder on a single figure
    # 'read':   read ROC data from the specified folder, then display the ROC curve
    # 'scroll': read the same datasets from more than one folder and
    #           scroll through them always showing results for the same dataset
    # 'chart':  create a table for all datasets and for all folders in SCROLL_FOLDERS showing auc values


    ####################
    # DATA PREPARATION #
    ####################
    if OPERATION == 'save' or OPERATION == 'draw':
        # read all filenames from specified directory, keep only the names, no directories
        tmp_files = glob.glob(DATA_FOLDER + '/*.csv')
        files = list()
        for tmp in tmp_files:
            files.append(tmp.split('\\')[-1])

        print('\nROC, AUC metrics generation for {} dataset(s) in ../{}/\n'.format(len(files), DATA_FOLDER))

    fileindex = 0  # we begin with the first file
    done = True if OPERATION in {'read', 'scroll', 'chart'} else False

    while not done:
        progress_for_executions = {}
        if THRESHOLD_INCREASE_MODE == 'corrected':
            thresholds = func.calculate_thresholds(THRESHOLD_STEPS, THRESHOLD_LIMIT, THRESHOLD_CENTER)
        elif THRESHOLD_INCREASE_MODE == 'fixed':
            # thresholds = [3]
            thresholds = [6, 5.5, 5, 4.5, 4,
                          3.833333333, 3.666666667, 3.5, 3.333333333, 3.166666667, 3, 2.833333333,
                          2.666666667, 2.5, 2.333333333, 2.166666667, 2, 1.833333333, 1.666666667,
                          1.5, 1, 0, -1, -6]

        # choose date and time, update fileindex if neccessary
        print('\n' + '#' * 100)
        print('# Generating ROC metrics for file {} of {} ({}).'.format(fileindex + 1, len(files), files[fileindex]))
        print('#' * 100 + '\n')

        # read the original dataset
        data_filename = DATA_FOLDER + '/' + files[fileindex]
        original_dataset = pd.read_csv(data_filename)['value']

        # read flags (ground truth)
        flags_filename = FLAGS_FOLDER + '/' + files[fileindex]
        ground_truth = pd.read_csv(flags_filename)['timestep']

        # create lists for the ROC curve points
        roc_dots = list()

        # beginning of concurrent environment
        # Manager is necessary to share objects with processes
        # ThreadPoolExecutor automatically allocates threads and joins them
        manager = multiprocessing.Manager()
        progress_for_executions = manager.dict(progress_for_executions)
        executions_max = manager.dict({})
        roc_dots = manager.list(roc_dots)
        for thsind in range(int(np.floor(THRESHOLD_STEPS/(multiprocessing.cpu_count() - 1)) + 1)):
            with multiprocessing.Pool() as executor:
                for cpuind in range(multiprocessing.cpu_count() - 1):
                    # Starting the execution for all steps
                    print('\n|' + '-' * 30)
                    print('| ReRe algorithm pass {} / {} '.format(thsind*(multiprocessing.cpu_count() - 1) + cpuind + 1,
                                                                  THRESHOLD_STEPS))

                    #############################
                    # ADJUST THRESHOLD STRENGTH #
                    #############################
                    if THRESHOLD_INCREASE_MODE == 'linear' or THRESHOLD_INCREASE_MODE == 'corrected':
                        if thsind == 0 and cpuind == 0:
                            threshold = (-1) * THRESHOLD_LIMIT  # we start with the lowest possible threshold
                        else:
                            threshold = func.threshold_adjuster(THRESHOLD_STEPS, THRESHOLD_INCREASE_MODE,
                                                                THRESHOLD_LIMIT, thresholds, threshold)

                    elif THRESHOLD_INCREASE_MODE == 'fixed':
                        threshold = func.threshold_adjuster(THRESHOLD_STEPS, THRESHOLD_INCREASE_MODE,
                                                            THRESHOLD_LIMIT, thresholds)

                    if threshold not in progress_for_executions:
                        progress_for_executions[threshold] = 0

                        print("| THRESHOLD_STRENGTH = {}".format(threshold))

                        ##########################################################################
                        # PUT INTO POOL THE EXECUTOR FUNCTION WITH CURRENT THRESHOLD,            #
                        # EXECUTION WILL START CONCURRENTLY AND THREADS WILL BE JOINED WHEN DONE #
                        # deepcopy to avoid race conditions                                      #
                        ##########################################################################
                        executor.apply_async(func.run_with_threshold, args=(threshold, data_filename, roc_dots,
                                                                            progress_for_executions, executions_max,
                                                                            deepcopy(ground_truth),
                                                                            deepcopy(ANOMALY_WINDOW_SIZE),
                                                                            deepcopy(NORMALIZE_ANOM_METRICS),
                                                                            deepcopy(ADJUST_SIGNALS),
                                                                            deepcopy(METRIC)))

                    else:
                        print("| Instance with the same THRESHOLD_STRENGTH already running, hence this won't run.")

                    print('|' + '-' * 30 + '\n')

                executor.close()
                executor.join()

        #############################################
        # SAVE AND DISPLAY ROC CURVE AND AUC METRIC #
        #############################################
        # creating an array for the parameters
        params = pd.DataFrame(index=range(2), columns=range(8))
        params[0][0] = 'THRESHOLD_STEPS'
        params[0][1] = THRESHOLD_STEPS
        params[1][0] = 'THRESHOLD_LIMIT'
        params[1][1] = THRESHOLD_LIMIT
        params[2][0] = 'THRESHOLD_CENTER'
        params[2][1] = THRESHOLD_CENTER
        params[3][0] = 'THRESHOLD_INCREASE_MODE'
        params[3][1] = THRESHOLD_INCREASE_MODE
        params[4][0] = 'ANOMALY_WINDOW_SIZE'
        params[4][1] = ANOMALY_WINDOW_SIZE
        params[5][0] = 'ADJUST_SIGNALS'
        params[5][1] = ADJUST_SIGNALS
        params[6][0] = 'NORMALIZE_ANOM_METRICS'
        params[6][1] = NORMALIZE_ANOM_METRICS
        params[7][0] = 'METRIC'
        params[7][1] = METRIC

        if OPERATION == 'save' or OPERATION == 'draw':
            func.saver(METRICS_FOLDER, files[fileindex], roc_dots, params)

        print('\n' + '#' * 38)
        print('# ROC, AUC metrics generated, saved. #')
        print('#' * 38 + '\n')

        done = fileindex == len(files) - 1
        fileindex += 1

    if OPERATION == 'draw' or OPERATION == 'read':
        # assembling filenames
        if OPERATION == 'draw' or OPERATION == 'read':
            if OPERATION == 'draw':
                plot_folder = '../' + METRICS_FOLDER + '/ROC/'
            else:
                plot_folder = '../' + READ_FOLDER + '/'
            func.plotter(plot_folder, ROUND, DRAW_THD_VALUES, ROC_LOOK_MODE)

    elif OPERATION == 'scroll':
        plot_folders = list()
        for i in range(len(SCROLL_FOLDERS)):
            plot_folders.append('../' + SCROLL_FOLDERS[i] + '/')
        func.scroll_plotter(plot_folders, ROUND, DRAW_THD_VALUES, ROC_LOOK_MODE)

    elif OPERATION == 'chart':
        func.caluclate_auc_chart(SCROLL_FOLDERS, ROC_LOOK_MODE, CHART_FOLDER)
