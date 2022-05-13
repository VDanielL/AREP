import csv
from copy import deepcopy


class PeriodogramCSVSaver:
    FILENAME = "periodogram_results.csv"

    def __init__(self, header):
        self.HEADER_ROW = deepcopy(header)
        self.HEADER_ROW.insert(0, "Dataset")
        with open(self.FILENAME, 'w', newline='') as file:
            csv.DictWriter(file, fieldnames=self.HEADER_ROW).writeheader()

    def write_row(self, dataset_name, results):
        row = {}
        results.insert(0, dataset_name)
        i = 0
        for idx in self.HEADER_ROW:
            row[idx] = results[i]
            i += 1
        with open(self.FILENAME, 'a', newline='') as file:
            csv.DictWriter(file, fieldnames=self.HEADER_ROW).writerow(row)


class ResultsCSVSaver:
    FILENAME = "decisions.csv"
    HEADER = ["Category", "Dataset", "Periodic", "Spiked", "Apt for AREP"]

    def __init__(self):
        with open(self.FILENAME, 'w', newline='') as file:
            csv.DictWriter(file, fieldnames=self.HEADER).writeheader()

    def write_row(self, category_name, dataset_name, results):
        row = {}
        results.insert(0, dataset_name)
        results.insert(0, category_name)
        i = 0
        for idx in self.HEADER:
            row[idx] = results[i]
            i += 1
        with open(self.FILENAME, 'a', newline='') as file:
            csv.DictWriter(file, fieldnames=self.HEADER).writerow(row)


class AllDataCSVSaver:
    FILENAME = 'all_data_type_data.csv'
    HEADER = ['Dataset', 'Idx of max', 'Periodic', 'Spike thd', 'Above ratio', 'Spiked', 'Length']

    def __init__(self):
        with open(self.FILENAME, 'w', newline='') as file:
            csv.DictWriter(file, fieldnames=self.HEADER).writeheader()

    def write_row(self, results):
        row = {}
        for i, col in enumerate(self.HEADER):
            row[col] = results[i]

        with open(self.FILENAME, 'a', newline='') as file:
            csv.DictWriter(file, fieldnames=self.HEADER).writerow(row)
