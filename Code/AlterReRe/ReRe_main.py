"""
MAIN FILE FOR THE RERE ALGORITHM

This file runs the improved algorithm based on ReRe.
Run this.
Set parameters in ReRe/ReRe.
"""
import multiprocessing


if __name__ == '__main__':
    # import components
    import ReRe.ReRe as ReRe
    import ReRe_main_functions as func

    if ReRe.ReRe.OPERATION == 'file':
        manager = multiprocessing.Manager()
        progress_for_executions = manager.dict({})
        executions_max = manager.dict({})
        with multiprocessing.Pool() as executor:
            # for loop for doing multiple passes using test-run files
            for test_instance in range(ReRe.ReRe.run_for):
                executor.apply_async(func.run_rere, args=(test_instance, progress_for_executions, executions_max))

            executor.close()
            executor.join()
    else:
        func.run_rere(1, {}, {})
