"""
STATUS BAR FUNCTION

This file contains the function necessary for drawing the status bar.
"""

import sys

bar_length = 50
fill_character = '█'
empty_character = ' '


def draw_batch_status_bar(progress_for_executions, executions_max, run_for):
    print('-' * 30)
    for key, value in progress_for_executions.items():
        percentage = round(100 * (value + 1) / executions_max[key], 2)
        block_number = int(round((value + 1) / executions_max[key] * bar_length))

        if run_for == 1:
            print("TS = {0:+.2f}:\t |{1}| {2}/{3} ({4} %)".format(
                key, block_number * fill_character + empty_character * (bar_length - block_number),
                value+1, executions_max[key], percentage))
        else:
            print("Execution {0:02d}/{1}:\t |{2}| {3}/{4} ({5} %)".format(
                key+1, run_for, block_number * fill_character + empty_character * (bar_length - block_number),
                value+1, executions_max[key], percentage))
    print('-' * 30)


def draw_status_bar(progress, max):
    percentage = round(100 * progress / max, 2)
    block_number = int(round(progress / max * bar_length))
    to_print = '\r|{0}| {1}/{2} ({3} %)'.format(
        block_number * fill_character + empty_character * (bar_length - block_number),
        progress+1, max, percentage)
    sys.stdout.write(to_print)
    sys.stdout.flush()
