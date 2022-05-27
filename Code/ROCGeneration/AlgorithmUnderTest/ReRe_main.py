"""
MAIN FILE FOR THE RERE ALGORITHM

This file contains the whole improved algorithm based on ReRe.
"""
import multiprocessing
import numpy as np


if __name__ == '__main__':
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    import ReRe.ReRe as ReRe
    import ReRe_main_functions as func

    if ReRe.ReRe.OPERATION == 'file':
        manager = multiprocessing.Manager()
        progress_for_executions = manager.dict({})
        executions_max = manager.dict({})

        outerlength = int(np.floor(ReRe.ReRe.run_for/(multiprocessing.cpu_count() - 1)) + 1)
        innerlength = multiprocessing.cpu_count() - 1

        for outeri in range(outerlength):
            with multiprocessing.Pool() as executor:
                # for loop for doing multiple passes using test-run files
                for inneri in range(innerlength):
                    test_instance = outeri * innerlength + inneri
                    if test_instance < ReRe.ReRe.run_for:
                        executor.apply_async(func.run_rere,
                                             args=(test_instance, progress_for_executions, executions_max))

                executor.close()
                executor.join()
    else:
        func.run_rere(1, {}, {})
