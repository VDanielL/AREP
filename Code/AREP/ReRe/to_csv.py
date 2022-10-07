"""
CSV FILE EXPORT FUNCTIONS

This file makes possible exporting ReRe data to .csv files.
"""

import time as timelib
import numpy as np
import csv
from datetime import datetime
from pathlib import Path
import os


def init_to_csv(self):
    if not self.EVAL_EXPORT:
        # creating file
        current_time = datetime.now().strftime('%Y%m%d%H%M%S')
        path_to_results = 'results'

        if not os.path.exists(path_to_results):
            os.mkdir(path_to_results)

        result_filename = f'{path_to_results}/{current_time}{self.inst_num}_results.csv'
        param_filename = f'{path_to_results}/{current_time}{self.inst_num}_hyperparams.csv'

        if self.USE_OFFSET_COMP:
            if self.USE_AUTOMATIC_OFFSET:
                if self.USE_AUTOMATIC_WS_AP:
                    data_names = ['timestep', 'original', 'predicted_1', 'predicted_2', 'AARE_1', 'AARE_2', 'thd_1',
                                  'thd_2', 'anomalies_1', 'anomalies_2', 'anomalies_aggr', 'pattern_change_1',
                                  'pattern_change_2', 'diff_avg_1', 'diff_avg_2', 'values_mean', 'pred_mean_1',
                                  'pred_mean_2', 'offset_retrain_threshold', 'offset_retrain_trigger_1',
                                  'offset_retrain_trigger_2', 'offset_percentage_1', 'offset_percentage_2',
                                  'retrain_percentage_1', 'retrain_percentage_2', 'window_size', 'age_power',
                                  'anom_flap', 'freq_sig', 'long_anom', 'no_sig', 'timesteps_dur']
                else:
                    data_names = ['timestep', 'original', 'predicted_1', 'predicted_2', 'AARE_1', 'AARE_2', 'thd_1',
                                  'thd_2', 'anomalies_1', 'anomalies_2', 'anomalies_aggr', 'pattern_change_1',
                                  'pattern_change_2', 'diff_avg_1', 'diff_avg_2', 'values_mean', 'pred_mean_1',
                                  'pred_mean_2', 'offset_retrain_threshold', 'offset_retrain_trigger_1',
                                  'offset_retrain_trigger_2', 'offset_percentage_1', 'offset_percentage_2',
                                  'retrain_percentage_1', 'retrain_percentage_2', 'timesteps_dur']

            else:
                if self.USE_AUTOMATIC_WS_AP:
                    data_names = ['timestep', 'original', 'predicted_1', 'predicted_2', 'AARE_1', 'AARE_2', 'thd_1',
                                  'thd_2', 'anomalies_1', 'anomalies_2', 'anomalies_aggr', 'pattern_change_1',
                                  'pattern_change_2', 'diff_avg_1', 'diff_avg_2', 'values_mean', 'pred_mean_1',
                                  'pred_mean_2', 'offset_retrain_threshold', 'offset_retrain_trigger_1',
                                  'offset_retrain_trigger_2', 'window_size', 'age_power', 'anom_flap', 'freq_sig',
                                  'long_anom', 'no_sig', 'timesteps_dur']
                else:
                    data_names = ['timestep', 'original', 'predicted_1', 'predicted_2', 'AARE_1', 'AARE_2', 'thd_1',
                                  'thd_2', 'anomalies_1', 'anomalies_2', 'anomalies_aggr', 'pattern_change_1',
                                  'pattern_change_2', 'diff_avg_1', 'diff_avg_2', 'values_mean', 'pred_mean_1',
                                  'pred_mean_2', 'offset_retrain_threshold', 'offset_retrain_trigger_1',
                                  'offset_retrain_trigger_2', 'timesteps_dur']
        else:
            if self.USE_AUTOMATIC_WS_AP:
                data_names = ['timestep', 'original', 'predicted_1', 'predicted_2', 'AARE_1', 'AARE_2', 'thd_1',
                              'thd_2', 'anomalies_1', 'anomalies_2', 'anomalies_aggr', 'pattern_change_1',
                              'pattern_change_2', 'window_size', 'age_power', 'anom_flap', 'freq_sig', 'long_anom',
                              'no_sig', 'timesteps_dur']
            else:
                data_names = ['timestep', 'original', 'predicted_1', 'predicted_2', 'AARE_1', 'AARE_2', 'thd_1',
                              'thd_2', 'anomalies_1', 'anomalies_2', 'anomalies_aggr', 'pattern_change_1',
                              'pattern_change_2', 'timesteps_dur']

        with open(result_filename, 'w', newline='') as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=data_names)
            csv_writer.writeheader()

        return data_names, param_filename, result_filename

    else:
        # determine algorithm
        algorithm = ''
        if not self.USE_WINDOW:
            algorithm = 'ReRe'
        elif not self.USE_OFFSET_COMP:
            algorithm = 'Alter-ReRe'
        else:
            algorithm = 'AREP'

        # determine category
        category = self.FILENAME.split('/')[2]

        # determine dataset name
        dataset = self.FILENAME.split('/')[-1]

        result_folder_path = self.EVAL_FOLDER + '/' + algorithm + '/' + category
        result_file_path = result_folder_path + '/' + algorithm + '_' + dataset

        # ensure directory exists
        Path(result_folder_path).mkdir(parents=True, exist_ok=True)

        with open(result_file_path, 'w', newline='') as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=['index', 'anomaly_score'])
            csv_writer.writeheader()

        return ['index', 'anomaly_score'], '', result_file_path


def dump_hyperparameters(self, self_start, param_filename):
    if not self.EVAL_EXPORT:
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
                'hyperparameter': 'USE_AUTOMATIC_WS_AP',
                'value': self.USE_AUTOMATIC_WS_AP,
                'description': 'Whether to use automatic setting of the WS and AP parameters.'
            }
            csv_writer__.writerow(row)
            row = {
                'hyperparameter': 'USE_OFFSET_COMP',
                'value': self.USE_OFFSET_COMP,
                'description': 'Whether to use offset compensation.'
            }
            csv_writer__.writerow(row)
            if self.USE_OFFSET_COMP:
                row = {
                    'hyperparameter': 'OFFSET_WINDOW_SIZE',
                    'value': self.OFFSET_WINDOW_SIZE,
                    'description': 'The number of values to use with the offset compensation algorithm.'
                }
                csv_writer__.writerow(row)
            if self.USE_OFFSET_COMP:
                row = {
                    'hyperparameter': 'OFFSET_PERCENTAGE',
                    'value': self.OFFSET_PERCENTAGE,
                    'description': 'The percentage of values to be above threshold to trigger LSTM retrain.'
                }
                csv_writer__.writerow(row)
            if self.USE_OFFSET_COMP:
                row = {
                    'hyperparameter': 'USE_AUTOMATIC_OFFSET',
                    'value': self.USE_AUTOMATIC_OFFSET,
                    'description': 'Whether to use automatic tuning of offset compensation parameters.'
                }
                csv_writer__.writerow(row)
            if self.USE_OFFSET_COMP:
                row = {
                    'hyperparameter': 'ACCEPTABLE_AVG_DURATION',
                    'value': self.ACCEPTABLE_AVG_DURATION,
                    'description': '[seconds]: the maximum allowed duration of one timestep on average.'
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
    if not self.EVAL_EXPORT:
        with open(result_filename, 'a') as csv_file_:
            csv_writer_ = csv.DictWriter(csv_file_, fieldnames=data_names)
            if self.USE_OFFSET_COMP:
                if self.USE_AUTOMATIC_OFFSET:
                    if self.USE_AUTOMATIC_WS_AP:
                        info = {
                            'timestep': time, 'original': self.values[time],
                            'predicted_1': float(self.predicted_1[time]), 'predicted_2': float(self.predicted_2[time]),
                            'AARE_1': self.AARE_1[time], 'AARE_2': self.AARE_2[time],
                            'thd_1': self.threshold_1[time], 'thd_2': self.threshold_2[time],
                            'anomalies_1': self.anomaly_1[time], 'anomalies_2': self.anomaly_2[time],
                            'anomalies_aggr': self.anomaly_aggr[time],
                            'pattern_change_1': self.pattern_change_1[time],
                            'pattern_change_2': self.pattern_change_2[time],
                            'diff_avg_1': self.diff_avg_1[time],
                            'diff_avg_2': self.diff_avg_2[time],
                            'values_mean': self.values_mean[time],
                            'pred_mean_1': self.pred_mean_1[time],
                            'pred_mean_2': self.pred_mean_2[time],
                            'offset_retrain_threshold': self.offset_retrain_threshold[time],
                            'offset_retrain_trigger_1': self.offset_retrain_trigger_1[time],
                            'offset_retrain_trigger_2': self.offset_retrain_trigger_2[time],
                            'offset_percentage_1': self.offset_percentage_1[time],
                            'offset_percentage_2': self.offset_percentage_2[time],
                            'retrain_percentage_1': self.retrain_percentage_1[time],
                            'retrain_percentage_2': self.retrain_percentage_2[time],
                            'window_size': self.WINDOW_SIZE, 'age_power': self.AGE_POWER,
                            'anom_flap': self.anom_flap[time], 'freq_sig': self.freq_sig[time],
                            'long_anom': self.long_anom[time], 'no_sig': self.no_sig[time],
                            'timesteps_dur': self.timesteps_dur[time]
                        }
                    else:
                        info = {
                            'timestep': time, 'original': self.values[time],
                            'predicted_1': float(self.predicted_1[time]), 'predicted_2': float(self.predicted_2[time]),
                            'AARE_1': self.AARE_1[time], 'AARE_2': self.AARE_2[time],
                            'thd_1': self.threshold_1[time], 'thd_2': self.threshold_2[time],
                            'anomalies_1': self.anomaly_1[time], 'anomalies_2': self.anomaly_2[time],
                            'anomalies_aggr': self.anomaly_aggr[time],
                            'pattern_change_1': self.pattern_change_1[time],
                            'pattern_change_2': self.pattern_change_2[time],
                            'diff_avg_1': self.diff_avg_1[time],
                            'diff_avg_2': self.diff_avg_2[time],
                            'values_mean': self.values_mean[time],
                            'pred_mean_1': self.pred_mean_1[time],
                            'pred_mean_2': self.pred_mean_2[time],
                            'offset_retrain_threshold': self.offset_retrain_threshold[time],
                            'offset_retrain_trigger_1': self.offset_retrain_trigger_1[time],
                            'offset_retrain_trigger_2': self.offset_retrain_trigger_2[time],
                            'offset_percentage_1': self.offset_percentage_1[time],
                            'offset_percentage_2': self.offset_percentage_2[time],
                            'retrain_percentage_1': self.retrain_percentage_1[time],
                            'retrain_percentage_2': self.retrain_percentage_2[time],
                            'timesteps_dur': self.timesteps_dur[time]
                        }
                else:
                    if self.USE_AUTOMATIC_WS_AP:
                        info = {
                            'timestep': time, 'original': self.values[time],
                            'predicted_1': float(self.predicted_1[time]), 'predicted_2': float(self.predicted_2[time]),
                            'AARE_1': self.AARE_1[time], 'AARE_2': self.AARE_2[time],
                            'thd_1': self.threshold_1[time], 'thd_2': self.threshold_2[time],
                            'anomalies_1': self.anomaly_1[time], 'anomalies_2': self.anomaly_2[time],
                            'anomalies_aggr': self.anomaly_aggr[time],
                            'pattern_change_1': self.pattern_change_1[time],
                            'pattern_change_2': self.pattern_change_2[time],
                            'diff_avg_1': self.diff_avg_1[time],
                            'diff_avg_2': self.diff_avg_2[time],
                            'values_mean': self.values_mean[time],
                            'pred_mean_1': self.pred_mean_1[time],
                            'pred_mean_2': self.pred_mean_2[time],
                            'offset_retrain_threshold': self.offset_retrain_threshold[time],
                            'offset_retrain_trigger_1': self.offset_retrain_trigger_1[time],
                            'offset_retrain_trigger_2': self.offset_retrain_trigger_2[time],
                            'window_size': self.WINDOW_SIZE, 'age_power': self.AGE_POWER,
                            'anom_flap': self.anom_flap[time], 'freq_sig': self.freq_sig[time],
                            'long_anom': self.long_anom[time], 'no_sig': self.no_sig[time],
                            'timesteps_dur': self.timesteps_dur[time]
                        }
                    else:
                        info = {
                            'timestep': time, 'original': self.values[time],
                            'predicted_1': float(self.predicted_1[time]), 'predicted_2': float(self.predicted_2[time]),
                            'AARE_1': self.AARE_1[time], 'AARE_2': self.AARE_2[time],
                            'thd_1': self.threshold_1[time], 'thd_2': self.threshold_2[time],
                            'anomalies_1': self.anomaly_1[time], 'anomalies_2': self.anomaly_2[time],
                            'anomalies_aggr': self.anomaly_aggr[time],
                            'pattern_change_1': self.pattern_change_1[time],
                            'pattern_change_2': self.pattern_change_2[time],
                            'diff_avg_1': self.diff_avg_1[time],
                            'diff_avg_2': self.diff_avg_2[time],
                            'values_mean': self.values_mean[time],
                            'pred_mean_1': self.pred_mean_1[time],
                            'pred_mean_2': self.pred_mean_2[time],
                            'offset_retrain_threshold': self.offset_retrain_threshold[time],
                            'offset_retrain_trigger_1': self.offset_retrain_trigger_1[time],
                            'offset_retrain_trigger_2': self.offset_retrain_trigger_2[time],
                            'timesteps_dur': self.timesteps_dur[time]
                        }
            else:
                if self.USE_AUTOMATIC_WS_AP:
                    info = {
                        'timestep': time, 'original': self.values[time],
                        'predicted_1': float(self.predicted_1[time]), 'predicted_2': float(self.predicted_2[time]),
                        'AARE_1': self.AARE_1[time], 'AARE_2': self.AARE_2[time],
                        'thd_1': self.threshold_1[time], 'thd_2': self.threshold_2[time],
                        'anomalies_1': self.anomaly_1[time], 'anomalies_2': self.anomaly_2[time],
                        'anomalies_aggr': self.anomaly_aggr[time],
                        'pattern_change_1': self.pattern_change_1[time],
                        'pattern_change_2': self.pattern_change_2[time],
                        'window_size': self.WINDOW_SIZE, 'age_power': self.AGE_POWER,
                        'anom_flap': self.anom_flap[time], 'freq_sig': self.freq_sig[time],
                        'long_anom': self.long_anom[time], 'no_sig': self.no_sig[time],
                        'timesteps_dur': self.timesteps_dur[time]
                    }
                else:
                    info = {
                        'timestep': time, 'original': self.values[time],
                        'predicted_1': float(self.predicted_1[time]), 'predicted_2': float(self.predicted_2[time]),
                        'AARE_1': self.AARE_1[time], 'AARE_2': self.AARE_2[time],
                        'thd_1': self.threshold_1[time], 'thd_2': self.threshold_2[time],
                        'anomalies_1': self.anomaly_1[time], 'anomalies_2': self.anomaly_2[time],
                        'anomalies_aggr': self.anomaly_aggr[time],
                        'pattern_change_1': self.pattern_change_1[time],
                        'pattern_change_2': self.pattern_change_2[time],
                        'timesteps_dur': self.timesteps_dur[time]
                    }

            csv_writer_.writerow(info)

    else:
        with open(result_filename, 'a') as csv_file_:
            csv_writer_ = csv.DictWriter(csv_file_, fieldnames=data_names)

            if np.isnan(self.anomaly_aggr[time]) or not (self.anomaly_aggr[time]):
                anomaly_score = 0
            else:
                anomaly_score = 1

            info = {
                'index': time,
                'anomaly_score': anomaly_score
            }
            csv_writer_.writerow(info)


def write_time(self, self_start, param_filename):
    if not self.EVAL_EXPORT:
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
