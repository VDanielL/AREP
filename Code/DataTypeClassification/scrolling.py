import glob
from msvcrt import getche
from matplotlib import pyplot as plt


# (re-)read the entire folder for all datasets
def scroll_read(datasets, directory):
    datasets.clear()
    files = glob.glob(directory + '/**/*.csv', recursive=True)
    for file in files:
        datasets.append(file)


# scrolling update at the end of the loop
def scroll_end(fileindex, directory, datasets, continuous):

    scroll_read(datasets, directory)

    if continuous:
        return False, fileindex + 1 if fileindex < len(datasets) - 1 else 0

    letterinfo = 'n: next, p: previous, #[0-9]: jump # forwards, -#[0-9]: jump # backwards, s: specific number, ' \
                 'e: escape, (default): reload '
    print(letterinfo)
    answer = getche().decode(errors='replace')

    plt.close('all')

    if answer == 'e':
        return True, 0
    elif answer == 'n':
        return False, fileindex + 1 if fileindex < len(datasets) - 1 else 0
    elif answer == 'p':
        return False, fileindex - 1 if fileindex > 0 else len(datasets) - 1
    elif '0' <= answer <= '9':
        return False, fileindex + int(answer) if fileindex < len(datasets) - int(answer) else \
            (int(answer) - (len(datasets) - fileindex)) % len(datasets)
    elif answer == '-':
        answer = getche().decode(errors='replace')
        if '0' <= answer <= '9':
            return False, fileindex - int(answer) if fileindex >= int(answer) else \
                len(datasets) - ((int(answer) - fileindex) % len(datasets))
        else:
            print('\n')
            return False, fileindex
    elif answer == 's':
        return False, int(input('pecify number (min 1, max {}): '.format(len(datasets)))) - 1
    else:
        print('\n')
        return False, fileindex


def separate_names(filename):
    if len(filename.split('\\')) == 3:
        category = filename.split('\\')[1]
        dataset = filename.split('\\')[2][:-4]  # no .csv at the end, only name
    else:
        category = ""
        dataset = filename.split('\\')[1]

    return category, dataset
