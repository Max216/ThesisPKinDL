import matplotlib.pyplot as plt
import numpy as np 
from collections import defaultdict

color_palette = [
'#e6194b', 
'#3cb44b', 
'#ffe119', 
'#0082c8', 
'#f58231', 
'#911eb4', 
'#46f0f0', 
'#000000', 
'#f032e6', 
'#d2f53c', 
'#fabebe', 
'#008080', 
'#aa6e28', 
'#800000', 
'#808000', 
'#000080', 
'#808080', 
'#e6beff',
'#aaffc3',
'#fffac8',
'#ff0000',
'#00ff00',
'#0000ff',
'#ff00ff',
'#234567',
'#00ffff'
]

def f7(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def plot_multi_bar_chart(data, title, legend_labels, save=None, rotate=0, width=0.1):
    '''
    Plot a bar chart with several bars per x value

    :param data     data to plot: [(label_x, [v1, v2, ...]), (...)]
    :param x_labels   x_labels
    :param title   title
    '''
    x_labels = [lbl for lbl, _ in data]
    data = [vals for _, vals in data]

    plot_data = [[] for i in range(len(data[0]))]
    for d in data:
        for i in range(len(d)):
            plot_data[i].append(d[i])


    num_groups = len(x_labels)

    fig, ax = plt.subplots()
    index = np.arange(num_groups)
    bar_width = width

    for i, lbl in enumerate(legend_labels):
        plt.bar(index + i * bar_width, plot_data[i], bar_width, color=color_palette[i], label=lbl)

    plt.xlabel('Gold label')
    plt.ylabel('# samples')
    plt.title(title)
    plt.xticks(index +  bar_width, x_labels, rotation=rotate)
    plt.legend(bbox_to_anchor=(0,1.12,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=2)
    plt.subplots_adjust(top=0.8)

    if save != None:
        print('TODO')
        1/0
    else:
        plt.show()



def plot_single_bar_chart(data, title, x_axis_name, y_axis_name, save=None):
    print(data)
    x_labels = [x for x,y in data]
    y_vals = [y for x,y in data]
    x_indizes = np.arange(len(data))
    width = 0.35
    color = [color_palette[i] for i in range(len(data))]

    plt.bar(x_indizes, y_vals, width, color=color)
    plt.ylabel(y_axis_name)
    plt.xlabel(x_axis_name)
    plt.xticks(x_indizes, x_labels)
    plt.title(title)

    plt.show()

def plot_double_chart_w_std(data1, data2, title, x_axis_name, y_axis_name, legend, block=True):
    lbls1 = [lbl for lbl, mean, std in data1]
    lbls2 = [lbl for lbl, mean, std in data2]
    x_labels = f7(list(lbls1 + lbls2))
    ind = np.arange(len(x_labels))
    width = 0.35

    fig, ax = plt.subplots()
    for i, data in enumerate([data1, data2]):
        mean_dict = defaultdict(int)
        sd_dict = defaultdict(int)
        for lbl, mean, sd in data:
            mean_dict[lbl] = mean
            sd_dict[lbl] = sd

        plot_means = [mean_dict[lbl] for lbl in x_labels]
        plot_stds = [sd_dict[lbl] for lbl in x_labels]

        chart = ax.bar(ind + i * width, plot_means, width, color=color_palette[i], yerr=plot_stds, label=legend[i])

    ax.set_ylabel(y_axis_name)
    ax.set_xlabel(x_axis_name)
    ax.set_title(title)
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(x_labels, rotation=90)
    plt.subplots_adjust(bottom=0.3)
    ax.legend(bbox_to_anchor=(0,1.12,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=2)
    plt.show(block=block)
    

