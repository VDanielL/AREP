"""
This file contains functions for generating ROC anomaly detection metrics.
These functions are called by 'immersive_ROC_AUC_metrics_generator'.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot
from datetime import datetime
from msvcrt import getche
import csv
import time as timelib
import glob
import pandas as pd
import multiprocessing

from AlgorithmUnderTest.ReRe.ReRe import ReRe

# LOCK FOR CONCURRENT EXECUTION
ROC_lock = multiprocessing.Lock()


# CALCULATE THRESHOLDS for ADJUSTED OPERATION
def calculate_thresholds(threshold_steps, threshold_limit, threshold_center):
    results = []

    if threshold_steps % 2 == 1:  # adjust for lower resources
        threshold_steps -= 1

    results.append(-threshold_limit)

    for i in range(threshold_steps // 2 - 1):  # calculate negative values
        results.append(results[i] / 2)
    for i in reversed(results):
        results.append(-i)
    results.insert(len(results) // 2, 0)  # insert 0 in the middle

    for i in range(len(results) - 2):
        results[i + 1] += threshold_center  # adjust for threshold_center

    results.pop(0)  # remove first element, since it will be chosen independently
    return results


# ADJUST THRESHOLD VALUES
def threshold_adjuster(thd_steps, increase_mode, sigmas_limit, thresholds, current=0):
    if increase_mode == 'linear':
        return current + (2 * sigmas_limit) / (thd_steps - 1)
    elif increase_mode == 'corrected' or increase_mode == 'fixed':
        if len(thresholds) > 0:
            return thresholds.pop(0)
        else:
            return 3.0


# RUN WITH A THRESHOLD STRENGTH FOR CONCURRENT OPERATION
def run_with_threshold(threshold_strength, data_filename, roc_dots, progress_for_executions, executions_max,
                       ground_truth, ANOMALY_WINDOW_SIZE, NORMALIZE_ANOM_METRICS, ADJUST_SIGNALS, METRIC):
    ############
    # RUN RERE #
    ############
    anomaly_detections = list()
    rere_runner(anomaly_detections, threshold_strength, data_filename, progress_for_executions, executions_max)

    ##################################
    # MEASURING THE CONFUSION MATRIX #
    ##################################
    TP, FP, TN, FN = confusion_matrix_measurer(anomaly_detections, ground_truth, ANOMALY_WINDOW_SIZE,
                                               NORMALIZE_ANOM_METRICS, ADJUST_SIGNALS)

    #######################################
    # CALCULATE NECESSARY ANOMALY METRICS #
    #######################################
    x, y = metrics_calculator(METRIC, TP, FP, TN, FN)
    with ROC_lock:
        roc_dots.append((x, y, threshold_strength))


# RUN BAREBONE RERE WITH ONLY PARTS NECESSARY FOR METRICS
def rere_runner(detections, threshold_strength, dataset_filename, progress_for_executions, executions_max):
    rere_instance = ReRe()

    # INITIALIZING PARAMETERS
    # this program will not modify key operational parameters, set them at
    # ReRe/initial.py at the manual parameter selection section!
    # IMPORTANT: ALSO SET OPERATION TO 'manual' in ReRe/ReRe.py, OTHERWISE ERRORS MIGHT OCCUR
    rere_instance.init_timer()
    rere_instance.init_offset_compensation()
    rere_instance.init_auto_offset_compensation()
    rere_instance.init_auto_ws_ap()

    rere_instance.OPERATION = 'manual'
    # the most important line below, tunes thd for ROC
    rere_instance.THRESHOLD_STRENGTH = threshold_strength
    rere_instance.inst_num = threshold_strength
    # hacking filename to open datasets from the ROC folder instead of the original intended 'datasets'
    # e.g. ../datasets/<filename>  <-> ../datasets/../forROC/<filename>
    rere_instance.FILENAME = dataset_filename
    # disable logging of parameters and values
    rere_instance.TO_CSV = False
    # disable debugging
    rere_instance.DEBUG = False
    # enable the status bar
    rere_instance.STATUS_BAR = True
    rere_instance.BATCH_STATUS_BAR = True
    rere_instance.progress_for_executions = progress_for_executions

    # load file, bring parameters to a consistent state
    rere_instance.load()
    rere_instance.initialize_cons()

    # for status bar
    rere_instance.executions_max = executions_max
    executions_max[rere_instance.inst_num] = rere_instance.length

    # preprocess the dataset
    rere_instance.preprocess()

    # ReRe algorithm starts now, starting timer
    timer_start = timelib.time()

    # initialize ReRe algorithm
    rere_instance.initialize_rere()

    # THE ACTUAL RERE ALGORITHM
    time = 0  # current timestep parameter
    while time < rere_instance.length:
        # start time measurement
        rere_instance.start_timestep()

        # update the beginning of the sliding window
        rere_instance.update_window_beginning(time)

        # perform one timestep of the original ReRe algorithm
        rere_instance.next_timestep(time)

        # extract the only valuable piece of information from ReRe: if there has been an anomaly signalled
        detections.append(rere_instance.anomaly_aggr[time])

        # perform offset compensation
        if rere_instance.USE_OFFSET_COMP:
            rere_instance.compensate_offset(time)

        # perform automatic tuning of offset compensation
        if rere_instance.USE_AUTOMATIC_OFFSET:
            rere_instance.auto_tune_offset(time)

        # perform automatic tuning of WINDOW_SIZE and AGE_POWER
        if rere_instance.USE_AUTOMATIC_WS_AP:
            rere_instance.auto_tune_ws_ap(time)

        # stop time measurement and update averages
        rere_instance.end_timestep(time)

        # jump to the next timestep
        time += 1

    return


# CALULATE AND RETURN THE AUC METRIC FOR A GIVEN CURVE OF X AND Y VALUES
def get_auc(xs, ys):
    auc = 0
    for t in range(1, len(xs)):
        auc += np.abs(xs[t - 1] - xs[t]) * np.min([np.abs(ys[t - 1]), np.abs(ys[t])]) + \
               .5 * np.abs(xs[t - 1] - xs[t]) * np.abs(ys[t - 1] - ys[t])
    return auc


# MEASURE THE CONFUSION MATRIX
def confusion_matrix_measurer(detections, ground_truth, aws, normalize, adjust):
    # (1) CREATE ANOMALY WINDOWS
    anomaly_windows = list([0] * len(detections))
    steps_since_middle = aws + 1
    overwrite_from = 0
    ground_truth_count = 0

    for y in range(len(detections)):
        # if we are at a timestep with a flagged anomaly:
        if np.count_nonzero(ground_truth == y) > 0:
            steps_since_middle += 1
            ground_truth_count += 1
            # if two windows conflict, overwrite previous window
            if steps_since_middle < aws:
                overwrite_from = int(y - np.floor((steps_since_middle - 1) / 2))
            else:
                overwrite_from = y - aws
            # don't overwrite negative indices
            overwrite_from = np.max([overwrite_from, 0])
            # overwrite array with the anomaly window
            for x in range(max(overwrite_from, 0), min(y + aws + 1, len(detections) - 1)):
                anomaly_windows[x] = ground_truth_count
            steps_since_middle = 0
        else:
            if ground_truth_count > 0:
                steps_since_middle += 1
    # if DEBUG:
    #     for y in time:
    #         print('{}: {}'.format(y, anomaly_windows[y]))

    # (2) ADJUST ANOMALY SIGNALS, CREATE ARRAY
    if adjust:
        anomalies_adjusted = list([False] * len(detections))
        # only the beginning of anomalies count as an anomaly signal (no->yes)
        for y in range(len(detections)):
            anomalies_adjusted[y] = detections[y] and (not detections[y - 1])
    else:
        anomalies_adjusted = detections

    # if DEBUG:
    #     for y in time:
    #         print('{}: {}'.format(y, anomalies_adjusted[y]))

    # (3) MEASURE CONFUSION MATRIX
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    last_detected_anomaly = 0

    for y in range(len(detections)):
        if y == 0:
            continue
        # if we are not in an anomaly window
        if anomaly_windows[y] == 0:
            # if we have just exited an anomaly window
            if anomaly_windows[y - 1] > 0:
                # if the last detected anomaly was not in the window we have just left
                if last_detected_anomaly != anomaly_windows[y - 1]:
                    false_negatives += 1
            # if there is an anomaly signal
            if anomalies_adjusted[y]:
                false_positives += 1
        # if we are inside an anomaly window
        else:
            # if the current window is different from the previous and
            # we haven't just entered an anomaly window from 0
            if anomaly_windows[y - 1] != anomaly_windows[y] and anomaly_windows[y - 1] != 0:
                # if the last detected anomaly was not in the window we have just left
                if last_detected_anomaly != anomaly_windows[y - 1]:
                    false_negatives += 1
            # if there is an anomaly signal
            if anomalies_adjusted[y]:
                # if the last detected anomaly was not in this window
                if last_detected_anomaly != anomaly_windows[y]:
                    true_positives += 1
                    last_detected_anomaly = anomaly_windows[y]
                    # if there is a true positive, calculate detection time as y_det - y_GT
                    # if DEBUG:
                    #     print('y: {}, GT: {}, y - GT: {}'.format(y, ground_truth[anomaly_windows[y] - 1],
                    #                                              y - ground_truth[anomaly_windows[y] - 1]))
    # normalize false positives and true negatives if selected
    if normalize:
        false_positives = false_positives / (2 * aws + 1)
        true_negatives = len(detections) / (2 * aws + 1) - \
                         true_positives - false_positives - false_negatives
    else:
        true_negatives = len(detections) - true_positives - false_positives - false_negatives

    return true_positives, false_positives, true_negatives, false_negatives


# CALCULATE NECESSARY METRICS FOR ROC
def metrics_calculator(settings, tp, fp, tn, fn):
    if settings == 'classic':
        # classic settings: X - FPR, Y - TPR

        # X - FPR
        if fp + tn == 0:
            fpr = 0
        else:
            fpr = fp / (fp + tn)

        # Y - TPR
        if tp + fn == 0:
            tpr = 0
        else:
            tpr = tp / (tp + fn)

        return fpr, tpr

    elif settings == 'prec_rec':
        # prec_rec settings: X - precision, Y - recall

        # X - precision
        if tp + fp != 0:
            precision = tp / (tp + fp)
        else:
            precision = 0

        # Y - recall
        if tp + fn != 0:
            recall = tp / (tp + fn)
        else:
            recall = 0

        return precision, recall


# SAVE ROC DATA TO FILES
def saver(data_folder, dataname, roc_dots, parameter_info):
    # ROC points might have been added out of order, restore this for integration
    roc_dots.sort()

    # assemble filenames
    savedate = datetime.now().strftime('%Y%m%d%H%M%S')
    save_filename_roc = '../' + data_folder + '/ROC/' + '{}_'.format(savedate) + dataname[0:-5] + '_ROC.csv'
    save_filename_param = '../' + data_folder + '/ROC/' + '{}_'.format(savedate) + dataname[0:-5] + '_param.csv'

    # assemble headers
    roc_headers = ['x', 'y', 'thd']
    param_headers = ['parameter', 'value']

    # write files
    # ROC
    with open(save_filename_roc, 'w', newline='') as roc_file:
        roc_writer = csv.DictWriter(roc_file, fieldnames=roc_headers)
        roc_writer.writeheader()

    with open(save_filename_roc, 'a') as roc_file_:
        roc_writer_ = csv.DictWriter(roc_file_, fieldnames=roc_headers)
        for x, y, thd in roc_dots:
            info = {
                'x': x,
                'y': y,
                'thd': thd
            }
            roc_writer_.writerow(info)

    # parameters
    with open(save_filename_param, 'w', newline='') as param_file:
        param_writer = csv.DictWriter(param_file, fieldnames=param_headers)
        param_writer.writeheader()

    with open(save_filename_param, 'a') as param_file_:
        param_writer_ = csv.DictWriter(param_file_, fieldnames=param_headers)
        for i in range(parameter_info.shape[1]):
            info = {
                'parameter': parameter_info[i][0],
                'value': parameter_info[i][1]
            }
            param_writer_.writerow(info)

    return


# PROCESS ROC DATA FOR SMOOTHING
def smooth_process_roc(xs, ys, thds, draw_thd, look_mode):
    if look_mode == 'raw':
        return xs, ys, thds

    elif look_mode == 'lin_re':
        prev_y = 0
        indices_to_delete = list()

        for i, y in enumerate(ys):
            if y >= prev_y:
                prev_y = y
            else:
                indices_to_delete.append(i)

        xs = np.delete(xs, indices_to_delete)
        ys = np.delete(ys, indices_to_delete)
        if draw_thd:
            thds = np.delete(thds, indices_to_delete)

        return xs, ys, thds

    elif look_mode == 'smooth':
        # first check if the monotonic increase rule is at all violated
        violated = False
        for i in range(len(ys) - 1):
            if ys[i + 1] < ys[i]:
                violated = True
                break

        if not violated:
            return xs, ys, thds

        # (1) DETERMINE START POINT
        # (the first lower point before the ROC monotonic increase rule is violated)
        for i in range(len(ys) - 1):
            if ys[i + 1] < ys[i]:
                violation_point = i
                break
        for i in range(violation_point - 1, -1, -1):
            if ys[i] < ys[violation_point]:
                start_point = i
                break

        # (2) DETERMINE END POINT
        # (the point after which the curve won't fall below the violation point)
        for i in range(len(ys) - 1, violation_point + 1, -1):
            if ys[i] < ys[violation_point]:
                end_point = i + 1
                break

        # (3) CALCULATE POINTS BETWEEN START AND END POINTS
        # (this has to be done while preserving the AUC)
        nonnegative = False
        changeable = False
        while not(nonnegative and changeable):
            # (3.1) calculate current AUC
            auc = get_auc(xs, ys)

            # (3.2) calculate A_diff, the sum of A_next (the following area compared to the start line)
            xstep = np.insert(xs, end_point, xs[end_point])
            ystep = list()
            for t in range(0, start_point + 1):
                ystep.append(ys[t])
            for t in range(start_point + 1, end_point):
                ystep.append(ys[start_point])
            ystep.append(ys[start_point])
            for t in range(end_point, len(ys)):
                ystep.append(ys[t])

            A_step = get_auc(xstep, ystep)
            A_diff = auc - A_step
            # for t in range(start_point, end_point):
            #     if ys[t+1] - ys[t] != 0:
            #         x_mid = ((xs[t+1] - xs[t]) / (ys[t+1] - ys[t])) * (ys[start_point] - ys[t]) + xs[t]
            #     else:
            #         x_mid = xs[t+1]
            #     A_diff += .5 * ((x_mid - xs[t]) * (ys[t] - ys[start_point]) +
            #                     (xs[t+1] - x_mid) * (ys[t+1] - ys[start_point]))
            nonnegative = A_diff >= 0

            A_max = .5 * (xs[start_point+1] - xs[start_point]) * (ys[end_point] - ys[start_point]) + \
                    (xs[end_point] - xs[start_point+1]) * (ys[end_point] - ys[start_point])
            changeable = A_max >= A_diff

            if not(nonnegative and changeable):
                start_point = start_point - 1 if start_point > 0 else 0
                continue

            # (3.3) calculate height variable
            h = A_diff / (xs[end_point] - .5 * xs[start_point] - .5 * xs[start_point+1])

            # (3.4) adjust y values based on the above calculations
            ysnew = list()
            for t in range(0, start_point + 1):
                ysnew.append(ys[t])
            for t in range(start_point + 1, end_point):
                ysnew.append(ys[start_point] + h)
            xs = np.insert(xs, end_point, xs[end_point])
            ysnew.append(ys[start_point] + h)
            for t in range(end_point, len(ys)):
                ysnew.append(ys[t])

        return xs, ysnew, thds

    else:
        return xs, ys, thds


# PLOT ROC DATA
def plotter(folder, roundto, draw_thd, look_mode):
    # gathering filenames
    files = glob.glob(folder + '*_ROC.csv')
    filenames = list()
    for tempfile in files:
        filenames.append(tempfile.split('\\')[-1])

    print('\nDisplaying ROC, AUC metrics for {} datasets in {}\n'.format(len(files), folder))

    # open figure for plots
    fig = pyplot.figure()

    # reading parameter info from file
    # THIS ASSUMES CONSTANT PARAMETERS OVER THE WHOLE FOLDER!
    parameters = pd.read_csv(folder + filenames[0][0:-8] + '_param.csv')
    roc_mode = parameters[parameters['parameter'] == 'METRIC'].to_numpy()[0][1]

    # display hyperparameters on figure as text
    param_text = ''
    for i in range(parameters.shape[0]):
        param_text += parameters['parameter'][i] + ': ' + parameters['value'][i] + ', '
        if i % 4 == 3:
            param_text += '\n'
    pyplot.text(.5, -.3, param_text, ha='center', linespacing=2)

    # iterating over all files and plotting their content
    for fileindex in range(len(files)):
        roc_x = pd.read_csv(folder + '/' + filenames[fileindex])['x'].to_numpy()
        roc_y = pd.read_csv(folder + '/' + filenames[fileindex])['y'].to_numpy()
        if draw_thd:
            thd = pd.read_csv(folder + '/' + filenames[fileindex])['thd'].to_numpy()

        # add the points (0,0) and (1,1) to the ROC curve, as per the recommendations in the book
        roc_x = np.insert(roc_x, 0, 0.0)
        roc_x = np.append(roc_x, 1.0)
        roc_y = np.insert(roc_y, 0, 0.0)
        roc_y = np.append(roc_y, 1.0)

        # smooth ROC curve if needed
        if draw_thd:
            roc_x, roc_y, thd = smooth_process_roc(roc_x, roc_y, thd, draw_thd, look_mode)
        else:
            roc_x, roc_y, _ = smooth_process_roc(roc_x, roc_y, '', draw_thd, look_mode)

        # the area under the curve is made of rectangles (first addendum) and triangles (second addendum)
        AUC = get_auc(roc_x, roc_y)

        p = pyplot.plot(roc_x, roc_y, marker='o', linestyle='-',
                        label=filenames[fileindex][15: -8] + ': AUC = {}'.format(round(AUC, roundto)))

        print(filenames[fileindex])

        # draw threshold values next to points
        if draw_thd:
            for i, txt in enumerate(thd):
                pyplot.gca().annotate(txt, (roc_x[i+1], roc_y[i+1]), xytext=(5, 5), textcoords='offset points',
                                      color=p[-1].get_color())
            # annotate manually added points
            pyplot.gca().annotate('added', (0, 0), xytext=(5, 5), textcoords='offset points', color=p[-1].get_color())
            pyplot.gca().annotate('added', (1, 1), xytext=(5, 5), textcoords='offset points', color=p[-1].get_color())

    # modifying the figure to look better
    title = 'Receiver Operating Characteristic curve (ROC)' if roc_mode == 'classic' else 'Precision-Recall Curve'
    xlabel = 'False Positive Rate' if roc_mode == 'classic' else 'Precision'
    ylabel = 'True Positive Rate' if roc_mode == 'classic' else 'Recall'

    pyplot.title(title)
    pyplot.xlabel(xlabel)
    pyplot.ylabel(ylabel)
    pyplot.legend()
    pyplot.grid()

    # move the axis up to leave room for the text
    pos1 = pyplot.gca().get_position()
    pos2 = [pos1.x0, pos1.y0 + .05, pos1.width, pos1.height]
    pyplot.gca().set_position(pos2)

    fig.set_size_inches(22 / 2.54, 18 / 2.54)  # :(
    pyplot.show()

    return


# CREATE GOOD LOOKING LABEL (change if necessary!)
def get_scroll_label(folder, dataset, auc, roundto):
    to_ret = ''
    foldername = folder.split('/')[-2]
    if foldername == 'ROCarep':
        to_ret += 'AREP'
    elif foldername == 'ROCalter':
        to_ret += 'Alter-Re$^2$'
    elif foldername == 'ROCrere':
        to_ret += 'ReRe'

    # to_ret += ' (' + dataset.split('\\')[-1][15: -8]
    # if dataset.split('\\')[-1][15: -8] == 'grok_asg_anomal':
    #     to_ret += 'y'
    # to_ret += '): AUC = ' + str(round(auc, roundto))
    return to_ret


# scroll through folders and compare results of two algorithms
def scroll_plotter(folders, roundto, draw_thd, look_mode):
    # function works by extracting the list of files from the first folder, then uses these datasets to open the same
    # results from the other folders on the list
    # gathering filenames
    first_files = glob.glob(folders[0] + '*_ROC.csv')
    first_filenames = list()
    for tempfile in first_files:
        first_filenames.append(tempfile.split('\\')[-1])

    print('\nScrolling through ROC, AUC metrics for {} datasets in {} folders\n'.format(len(first_files), len(folders)))

    fileindex = 0
    while True:
        # scrolling update
        fileindex = fileindex % len(first_filenames)
        dataset_name = first_filenames[fileindex][15:]
        print('Showing file {} of {} ({}).'.format(fileindex + 1, len(first_filenames), dataset_name))

        # open figure for plots
        fig = pyplot.figure()

        # reading parameter info from the file in THE FIRST FOLDER
        # THIS ASSUMES CONSTANT PARAMETERS OVER ALL FOLDERS
        parameters = pd.read_csv(folders[0] + first_filenames[0][0:-8] + '_param.csv')
        roc_mode = parameters[parameters['parameter'] == 'METRIC'].to_numpy()[0][1]

        # display hyperparameters on figure as text
        param_text = ''
        for i in range(parameters.shape[0]):
            param_text += parameters['parameter'][i] + ': ' + parameters['value'][i] + ', '
            if i % 4 == 3:
                param_text += '\n'
        # pyplot.text(.5, -.3, param_text, ha='center', linespacing=2)

        # iterating over all folders and plotting their content for the current dataset
        for i in range(len(folders)):
            # search for the name of the dataset in the current folder
            in_current_folder = glob.glob(folders[i] + '*' + dataset_name)
            if len(in_current_folder) == 0:
                continue
            # if there is at least one file of the dataset in the current folder
            else:
                for j in range(len(in_current_folder)):
                    current = in_current_folder[j]
                    roc_x = pd.read_csv(current)['x'].to_numpy()
                    roc_y = pd.read_csv(current)['y'].to_numpy()
                    if draw_thd:
                        thd = pd.read_csv(current)['thd'].to_numpy()

                    # add the points (0,0) and (1,1) to the ROC curve, as per the recommendations in the book
                    roc_x = np.insert(roc_x, 0, 0.0)
                    roc_x = np.append(roc_x, 1.0)
                    roc_y = np.insert(roc_y, 0, 0.0)
                    roc_y = np.append(roc_y, 1.0)

                    # smooth ROC curve if needed
                    if draw_thd:
                        roc_x, roc_y, thd = smooth_process_roc(roc_x, roc_y, thd, draw_thd, look_mode)
                    else:
                        roc_x, roc_y, _ = smooth_process_roc(roc_x, roc_y, '', draw_thd, look_mode)

                    # the area under the curve is made of rectangles (first addendum) and triangles (second addendum)
                    AUC = get_auc(roc_x, roc_y)

                    p = pyplot.plot(roc_x, roc_y, marker='o', linestyle='-', clip_on=False, color=['b', 'r', 'g'][i],
                                    label=get_scroll_label(folders[i], current, AUC, roundto))
                                    # label=folders[i].split('/')[-2] + ': ' + current.split('\\')[-1][15: -8] +
                                    #       ': AUC = {}'.format(round(AUC, roundto)))

                    # draw threshold values next to points
                    if draw_thd:
                        for k, txt in enumerate(thd):
                            pyplot.gca().annotate(txt, (roc_x[k], roc_y[k]), xytext=(5, 5), textcoords='offset points',
                                                  color=p[-1].get_color())
                        # annotate manually added points
                        pyplot.gca().annotate('added', (0, 0), xytext=(5, 5), textcoords='offset points',
                                              color=p[-1].get_color())
                        pyplot.gca().annotate('added', (1, 1), xytext=(5, 5), textcoords='offset points',
                                              color=p[-1].get_color())

        # modifying the figure to look better
        title = 'Receiver Operating Characteristic curve (ROC)' if roc_mode == 'classic' else 'Precision-Recall Curve'
        xlabel = 'False Positive Rate' if roc_mode == 'classic' else 'Precision'
        ylabel = 'True Positive Rate' if roc_mode == 'classic' else 'Recall'

        # pyplot.title(title)
        pyplot.xlabel(xlabel)
        pyplot.ylabel(ylabel)
        pyplot.legend(loc='lower right')
        pyplot.grid()

        pyplot.xlim([0, 1])
        pyplot.ylim([0, 1])

        # move the axis up to leave room for the text
        pos1 = pyplot.gca().get_position()
        pos2 = [pos1.x0 + .05, pos1.y0 + .05, pos1.width, pos1.height]
        pyplot.gca().set_position(pos2)

        # fig.set_size_inches(22 / 2.54, 18 / 2.54)  # :(
        fig.set_size_inches(10 / 2.54, 10 / 2.54)  # :(
        pyplot.show()

        ##########################
        # END TEXT FOR SCROLLING #
        ##########################
        # gathering filenames
        first_files = glob.glob(folders[0] + '*_ROC.csv')
        first_filenames = list()
        for tempfile in first_files:
            first_filenames.append(tempfile.split('\\')[-1])

        letterinfo = 'n: next, p: previous, #[0-9]: jump # forwards, -#[0-9]: jump # backwards, s: specific number, ' \
                     'e: escape, (default): reload '
        print(letterinfo)
        answer = getche().decode(errors='replace')
        pyplot.close('all')
        if answer == 'e':
            break
        elif answer == 'n':
            fileindex = fileindex + 1 if fileindex < len(first_filenames) - 1 else 0
        elif answer == 'p':
            fileindex = fileindex - 1 if fileindex > 0 else len(first_filenames) - 1
        elif '0' <= answer <= '9':
            fileindex = fileindex + int(answer) if fileindex < len(first_filenames) - int(answer) else \
                (int(answer) - (len(first_filenames) - (fileindex))) % len(first_filenames)
        elif answer == '-':
            answer = getche().decode(errors='replace')
            if '0' <= answer <= '9':
                fileindex = fileindex - int(answer) if fileindex >= int(answer) else \
                    len(first_filenames) - ((int(answer) - fileindex) % len(first_filenames))
            else:
                print('\n')
                continue
        elif answer == 's':
            fileindex = int(input('pecify number (min 1, max {}): '.format(len(first_filenames)))) - 1
        else:
            print('\n')
            continue

        print('\n')

    return


# GENERATE AUC RESULTS OUTPUT
def caluclate_auc_chart(folders, look_mode, chart_folder):
    dataset_paths = glob.glob('../' + folders[0] + '/*_param.csv')
    datasets = list()
    for da in dataset_paths:

        datasets.append(da.replace('\\', '/').split('/')[2 + folders[0].count('/')][15:-10])

    auc_results = pd.DataFrame(columns=folders, index=datasets)

    for folder in folders:
        for data in datasets:
            currents = glob.glob('../' + folder + '/*' + data + '_ROC.csv')
            if len(currents) == 0:
                print('PROBLEM: no dataset \'{}\' in folder \'{}\''.format(data, folder))
                continue
            elif len(currents) > 1:
                print('PROBLEM: more than one dataset \'{}\' in folder \'{}\', showing first'.format(data, folder))

            roc_x = pd.read_csv(currents[0])['x'].to_numpy()
            roc_y = pd.read_csv(currents[0])['y'].to_numpy()

            roc_x = np.insert(roc_x, 0, 0.0)
            roc_y = np.insert(roc_y, 0, 0.0)
            roc_x = np.append(roc_x, 1.0)
            roc_y = np.append(roc_y, 1.0)

            roc_x, roc_y, _ = smooth_process_roc(roc_x, roc_y, '', False, look_mode)

            auc_results[folder][data] = get_auc(roc_x, roc_y)

    auc_results.to_csv('../' + chart_folder + '/auc_results.csv')