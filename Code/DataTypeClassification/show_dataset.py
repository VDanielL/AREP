import numpy as np


def plot_original(fig, ax, values, flags, dataset_name, normal_mode):
    time = list()
    for i in range(len(values)):
        time.append(i)
    flag_plot = [np.NaN] * len(values)
    for flagindex in flags:
        flag_plot[flagindex] = values[flagindex]

    if normal_mode:
        fig.suptitle('Data type categories for \'' + dataset_name + '\'', fontsize=14, fontweight='bold')
    ax.plot(time, values, 'g')
    ax.plot(time, flag_plot, color='red', marker='o', markersize=10, linestyle='', label='maximum')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Value')
    ax.set_title('Original dataset' if normal_mode else dataset_name)
    ax.grid()


def param_result_text(fig, axs, maxindex_num, maxindex_thd, periodic, perc, perc_thd, spiked):
    # move axes to leave space for text
    for i in range(len(axs)):
        pos_o = axs[i].get_position()
        pos_n = [pos_o.x0 - .05, pos_o.y0 + (1 - i) * .03, .95 * pos_o.width, .85 * pos_o.height]
        axs[i].set_position(pos_n)

    # figure out positions for texts
    xp = .906
    yp_per_text = (axs[1].get_position().ymax + axs[1].get_position().ymin) / 2 + .03
    yp_per_res = (axs[1].get_position().ymax + axs[1].get_position().ymin) / 2 - .03
    yp_spi_text = (axs[2].get_position().ymax + axs[2].get_position().ymin) / 2 + .03
    yp_spi_res = (axs[2].get_position().ymax + axs[2].get_position().ymin) / 2 - .03

    # periodogram text
    fig.text(xp, yp_per_text,
             'index of max: ' + str(maxindex_num) + '\n' +
             'index threshold: ' + str(maxindex_thd),
             ha='center', linespacing=2)

    # periodogram result
    fig.text(xp, yp_per_res,
             ('PERIODIC' if periodic else 'APERIODIC'),
             ha='center', linespacing=2, fontweight='bold', size='large')

    # spikedness check text
    fig.text(xp, yp_spi_text,
             'above percentage: ' + str(perc) + '%\n' +
             'percentage threshold: ' + str(perc_thd) + '%',
             ha='center', linespacing=2)

    # spikedness result
    fig.text(xp, yp_spi_res,
             ('SPIKED' if spiked else 'NOT SPIKED'),
             ha='center', linespacing=2, fontweight='bold', size='large')

    # change the size of the figure when shown
    fig.set_size_inches(30/2.54, 18/2.54)  # :(


def text_resize_raw_mode(fig, ax):
    # move axis up to leave space for source text
    pos_o = ax.get_position()
    pos_n = [pos_o.x0, pos_o.y0 + .06, pos_o.width, .95*pos_o.height]
    ax.set_position(pos_n)

    # source text
    fig.text(.95, .05, 'Source of dataset: NAB.', ha='right', size='x-small')

    # change the size of the figure when shown
    fig.set_size_inches(30 / 2.54, 15 / 2.54)  # :(


