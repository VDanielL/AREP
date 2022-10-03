"""
CSV FILE EXPORT FUNCTIONS

This file makes possible exporting ReRe data to .csv files.
"""


import time as timelib
import csv
from datetime import datetime


def init_to_csv(self):
    # creating file
    current_time = datetime.now().strftime('%Y%m%d%H%M%S')
    result_filename = 'results/{}{}_results.csv'.format(current_time, self.inst_num)
    param_filename = 'results/{}{}_hyperparams.csv'.format(current_time, self.inst_num)
    data_names = ['timestep', 'original', 'predicted_1', 'predicted_2', 'AARE_1', 'AARE_2', 'thd_1', 'thd_2',
                  'anomalies_1', 'anomalies_2', 'anomalies_aggr', 'pattern_change_1', 'pattern_change_2',
                  'timesteps_dur']

    with open(result_filename, 'w', newline='') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=data_names)
        csv_writer.writeheader()

    return data_names, param_filename, result_filename


def dump_hyperparameters(self, self_start, param_filename):
    with open(param_filename, 'w', newline='') as csv_file__:
        csv_writer__ = csv.DictWriter(csv_file__,
                                      fieldnames=['hyperparameter',
                                                  'value',
                                                  'description'])
        csv_writer__.writeheader()
        row = {
            'hyperparameter': 'algorithm',
            'value': 'ReRe',
            'description': ''
        }
        csv_writer__.writerow(row)
        row = {
            'hyperparameter': 'B',
            'value': self.B,
            'description': 'Look-back param. of RePAD.'
        }
        csv_writer__.writerow(row)
        row = {
            'hyperparameter': 'F',
            'value': self.F,
            'description': 'Predict-forward param. of RePAD.'
        }
        csv_writer__.writerow(row)
        row = {
            'hyperparameter': 'THRESHOLD_STRENGTH',
            'value': self.THRESHOLD_STRENGTH,
            'description': 'Coefficient of sigma when calculating threshold.'
        }
        csv_writer__.writerow(row)
        row = {
            'hyperparameter': 'USE_WINDOW',
            'value': self.USE_WINDOW,
            'description': 'Whether to use window-mode execution.'
        }
        csv_writer__.writerow(row)
        if self.USE_WINDOW:
            row = {
                'hyperparameter': 'WINDOW_SIZE',
                'value': self.WINDOW_SIZE,
                'description': 'The size of the window when calculating AARE and thd values.'
            }
            csv_writer__.writerow(row)
        row = {
            'hyperparameter': 'USE_AGING',
            'value': self.USE_AGING,
            'description': 'Whether to use aging when calculating AARE.'
        }
        csv_writer__.writerow(row)
        if self.USE_AGING:
            row = {
                'hyperparameter': 'AGE_POWER',
                'value': self.AGE_POWER,
                'description': 'The power to which the calculated linear aging coefficient is raised.'
            }
            csv_writer__.writerow(row)
        row = {
            'hyperparameter': 'NUM_EPOCHS',
            'value': self.NUM_EPOCHS,
            'description': 'Number of epochs when training the LSTM model.'
        }
        csv_writer__.writerow(row)
        row = {
            'hyperparameter': 'NUM_NEURONS',
            'value': self.NUM_NEURONS,
            'description': 'Number of neurons of the LSTM model.'
        }
        csv_writer__.writerow(row)
        row = {
            'hyperparameter': 'FILENAME',
            'value': self.FILENAME,
            'description': 'The name of the file containing input data.'
        }
        csv_writer__.writerow(row)
        row = {
            'hyperparameter': 'DO_DIV',
            'value': self.DO_DIV,
            'description': 'Whether to divide data by the maximum times 1,1.'
        }
        csv_writer__.writerow(row)
        row = {
            'hyperparameter': 'USE_LESS',
            'value': self.USE_LESS,
            'description': 'Whether to use only the beginning of the dataset.'
        }
        csv_writer__.writerow(row)
        if self.USE_LESS:
            row = {
                'hyperparameter': 'LESS',
                'value': self.LESS,
                'description': 'The number of data point to use from the beginning.'
            }
            csv_writer__.writerow(row)
        row = {
            'hyperparameter': 'DEBUG',
            'value': self.DEBUG,
            'description': 'Whether to print debug info to the console.'
        }
        csv_writer__.writerow(row)
        row = {
            'hyperparameter': 'NOTES',
            'value': self.NOTES,
            'description': 'Any further notes.'
        }
        csv_writer__.writerow(row)
        row = {
            'hyperparameter': 'TOTAL_TIME',
            'value': 0,
            'description': 'Total time spent in seconds.'
        }
        csv_writer__.writerow(row)
        row = {
            'hyperparameter': 'start_time',
            'value': self_start,
            'description': 'INTERNAL PARAMETER FOR LIVE PLOTTER!'
        }
        csv_writer__.writerow(row)
        row = {
            'hyperparameter': 'full_size',
            'value': self.length,
            'description': 'INTERNAL PARAMETER FOR LIVE PLOTTER!'
        }
        csv_writer__.writerow(row)


def dump_results(self, time, data_names, result_filename):
    with open(result_filename, 'a') as csv_file_:
        csv_writer_ = csv.DictWriter(csv_file_, fieldnames=data_names)
        info = {
            'timestep': time, 'original': self.values[time],
            'predicted_1': float(self.predicted_1[time]), 'predicted_2': float(self.predicted_2[time]),
            'AARE_1': self.AARE_1[time], 'AARE_2': self.AARE_2[time],
            'thd_1': self.threshold_1[time], 'thd_2': self.threshold_2[time],
            'anomalies_1': self.anomaly_1[time], 'anomalies_2': self.anomaly_2[time],
            'anomalies_aggr': self.anomaly_aggr[time],
            'pattern_change_1': self.pattern_change_1[time],
            'pattern_change_2': self.pattern_change_2[time],
        }

        csv_writer_.writerow(info)


def write_time(self, self_start, param_filename):
    # write actual time elapsed info to csv
    with open(param_filename) as paramfin:
        reader = csv.reader(paramfin.readlines())

    with open(param_filename, 'w', newline='') as paramfout:
        writer = csv.writer(paramfout)
        for line in reader:
            if line[0] == 'TOTAL_TIME':
                writer.writerow([line[0], timelib.time() - self_start, 'Total time spent in seconds.'])
                break
            else:
                writer.writerow(line)
        writer.writerows(reader)