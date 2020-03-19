import os
import numpy as np
import random
import matplotlib.pyplot as plt

from collections import OrderedDict

import json
import pickle
import itertools


cmaps = OrderedDict()
cmaps['Sequential'] = [
            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
cmaps['Sequential-my'] = [
            'Reds', 'Blues', 'Greens', 'Oranges', 'Purples', 'PuRd', 'GnBu', 'BuGn']

def create_plot_labels():
    pass

def select_relevant_exps():
    pass

def plot_simple_run(data, labels, fig_size=(10,5)):

    n_points = 100

    if data == None:
        data_points = {}
        data_points['bce-loss/train'] = np.random.rand(n_points) * 2
        data_points['bce-loss/test'] = np.random.rand(n_points) * 2

        #data_points['bce-log-loss/train'] = np.random.rand(n_points) * 2
        #data_points['bce-log-loss/test'] = np.random.rand(n_points) * 2

    fig, ax = plt.subplots(figsize=fig_size)
    plt.style.use('seaborn')

    #color = iter(plt.cm.rainbow(np.linspace(0, 1, len(data_points))))
    #c = next(color)
    c = 'blue'
    for key, vals in data_points.items():
        if key.__contains__('test'):
            ls = '--'
        else:
            ls = '-'
        ax.plot(vals, label=key, linestyle=ls)

    ax.set_xlabel('Epoch')  # Add an x-label to the axes.
    ax.set_ylabel('y label')  # Add a y-label to the axes.
    ax.set_title("Simple Plot")  # Add a title to the axes.
    ax.legend()  # Add a legend.

    plt.show()

def plot_experiments(exp_data, exp_labels : [str], metric_first_plot : [str], sub_plots=(1, 2), fig_size=(12, 6)):

    '''
    input: list of dicts
    each list entry (experiment) contains losses (train + test) and metrics (4)
    corresponding metrics have same color but test is dashed

    data format: list(dict) [exp1, exp2, ... ]

    exp1:
    {   metric1 : (train, test)},
        metric2 : (train, test)},
        ... }

    exp2:
    {   metric1 : (train, test)},
        metric2 : (train, test)},
        ... }

    - From list of Sequential Colormaps Select n_experiments cmaps
    - each cmap n_metrics colors

    :return:
    '''
    if False:
        exp_data = []
        exp_labels = []
        metrics = ["loss", "acc", "ap"]
        metric_first_plot = ['loss']

        n = 10
        for i in range(3):
            exp_data.append({key: (np.random.rand(n), np.random.rand(n)) for key in metrics})
            exp_labels.append("exp " + str(i))

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=fig_size)

    n_exps = len(exp_data)
    #seq_cmaps = random.sample(cmaps['Sequential'], n_exps)
    seq_cmaps = cmaps['Sequential-my'][:n_exps]
    for i, exp in enumerate(exp_data):
        #select color map
        cmap = plt.get_cmap(seq_cmaps[i])
        n_metrics = len(exp.keys())
        #create colors for each metric
        colors = iter(cmap(np.linspace(0.5, 0.9, n_metrics)))
        for key in exp.keys():
            c = next(colors)
            train_vals, test_vals = exp[key]
            lbl = key

            if key in metric_first_plot:
                ax = ax1
                lbl = ' '.join([exp_labels[i], lbl])
            else:
                ax = ax2

            ax.plot(train_vals, label=lbl, color=c, linestyle='-')
            ax.plot(test_vals, label=lbl + '/test', color=c, linestyle='--')

    ax1.set_ylabel('value')
    ax1.set_xlabel('step')
    ax1.set_title('Loss')
    ax1.legend()

    ax2.set_xlabel('step')
    ax2.set_title('Performance Metrics')
    ax2.legend()

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    #plot_simple_run(None, None)
    plot_experiments(exp_data=None, exp_labels=['a', 'b', 'c'], metric_first_plot=['loss'])