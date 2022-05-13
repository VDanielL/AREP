import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
import json
import scrolling
import periodogram
import show_dataset
import to_csv
import spikedness

if __name__ == "__main__":

    ##############
    # PARAMETERS #
    ##############

    DIRECTORY_NAME = 'data'

    SIGMA_COEFFICIENT = 6  # spike threshold is calculated by: avg + SIGMA_COEFFICIENT * std
    THRESHOLD_PERCENTAGE = .4  # if the percentage of above thd spikes is larger than this, the dataset is spiked
    MAX_INDEX = 20  # the maximum allowed index before which the dataset can be considered aperiodic

    CONTINUOUS_SCROLLING = True  # after closing the figure for a dataset, open the next one straight away

    MODE = 'scroll'
    # 'scroll': scroll through all datasets, show plots, decide periodicity
    # 'chart':  use the array specified below to make all decisions for all MAX_INDEX parameter values, export to csv
    # 'decision_table': export decision made by the algorithm to csv file
    # 'all_data': export all information on all datasets
    # 'scroll_raw': scroll through all datasets, but only display them and the places of anomalies

    MAXINDEXES = [0, 1, 5, 10, 15, 20, 30, 50, 100]

    USE_LESS = False  # only use a part of the dataset, specified by LESS below
    LESS = 400  # the number of dataset datapoints to use

    #####################
    # END OF PARAMETERS #
    #####################

    datasets = list()
    fileindex = 0

    scrolling.scroll_read(datasets, DIRECTORY_NAME)

    done = True if MODE in {'chart', 'decision_table', 'all_data'} else False

    while not done:
        # scrolling update
        fileindex = fileindex % len(datasets)
        category_name, dataset_name = scrolling.separate_names(datasets[fileindex])
        print('Showing file {} of {} ({}).'.format(fileindex + 1, len(datasets), dataset_name))

        # read dataset and anomaly flags
        data = pd.read_csv(datasets[fileindex])#, parse_dates=['timestamp'])
        values = data['value'].tolist()
        time = data['timestamp'].tolist()
        with open('combined_labels.json') as json_file:
            labels = json.load(json_file)[category_name + '/' + dataset_name + '.csv']
        flag_indexes = list()
        for label in labels:
            flag_indexes.append(time.index(label))

        if USE_LESS:
            values = values[:LESS]
            time = time[:LESS]

        ####################
        # GENERATING PLOTS #
        ####################

        fig, axs = plt.subplots(3 if MODE != 'scroll_raw' else 1)

        # Plot of dataset
        if MODE != 'scroll_raw':
            dat_axs = axs[0]
        else:
            dat_axs = axs
        show_dataset.plot_original(fig, dat_axs, values, flag_indexes, dataset_name, MODE != 'scroll_raw')

        if MODE != 'scroll_raw':
            # Plot of periodogram
            periodic, maxi_num = periodogram.periodogram(values, axs[1], MAX_INDEX)

            # Plot of spike checker
            spiked, perc = spikedness.draw_spiked(values, axs[2], SIGMA_COEFFICIENT, THRESHOLD_PERCENTAGE)

            # Text of parameters and results, show plot
            show_dataset.param_result_text(fig, axs, maxi_num, MAX_INDEX, periodic, perc, THRESHOLD_PERCENTAGE, spiked)
        else:
            show_dataset.text_resize_raw_mode(fig, dat_axs)
        plt.show()

        ##########################
        # END TEXT FOR SCROLLING #
        ##########################

        over, fileindex = scrolling.scroll_end(fileindex, DIRECTORY_NAME, datasets, CONTINUOUS_SCROLLING)
        if over:
            break
        else:
            print('\n')

    if MODE == 'chart':
        # initialize csv wirter
        csv_saver = to_csv.PeriodogramCSVSaver(MAXINDEXES)

        for dataset in datasets:
            decisions = list()
            for maxi in MAXINDEXES:
                # read dataset
                data = pd.read_csv(dataset, parse_dates=['timestamp'])
                values = data['value'].to_numpy()
                time = data['timestamp'].to_numpy()

                if USE_LESS:
                    values = values[:LESS]
                    time = time[:LESS]

                f, Pxx_den = signal.periodogram(values)
                if periodogram.is_aperiodic(Pxx_den, maxi):
                    decisions.append('A')
                else:
                    decisions.append('P')

            csv_saver.write_row(dataset, decisions)

    if MODE == 'decision_table':
        csv_saver = to_csv.ResultsCSVSaver()
        for dataset in datasets:
            data = pd.read_csv(dataset, parse_dates=['timestamp'])
            values = data['value'].to_numpy()
            time = data['timestamp'].to_numpy()
            category_name, dataset_name = scrolling.separate_names(dataset)
            f, Pxx_den = signal.periodogram(values)
            aperiodic = periodogram.is_aperiodic(Pxx_den, MAX_INDEX)
            spiked = spikedness.check_spiked(values, SIGMA_COEFFICIENT, THRESHOLD_PERCENTAGE)[0]
            apt_for_AREP = aperiodic and not spiked
            # if apt_for_AREP:
            csv_saver.write_row(category_name, dataset_name, [not aperiodic, spiked, apt_for_AREP])

    if MODE == 'all_data':
        csv_saver = to_csv.AllDataCSVSaver()

        for dataset in datasets:
            # read dataset
            data = pd.read_csv(dataset, parse_dates=['timestamp'])
            values = data['value'].to_numpy()
            _, dataset_name = scrolling.separate_names(dataset)
            f, Pxx_den = signal.periodogram(values)
            aperiodic = periodogram.is_aperiodic(Pxx_den, MAX_INDEX)
            idx_of_max = np.argmax(Pxx_den) / (len(Pxx_den) - 1)
            spiked, time, _, thd, above_ratio = spikedness.check_spiked(values, SIGMA_COEFFICIENT, THRESHOLD_PERCENTAGE)
            csv_saver.write_row([dataset_name, idx_of_max, not aperiodic, thd[0], above_ratio, spiked, len(values)])




