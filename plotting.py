import matplotlib.pyplot as plt
import numpy as np 

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

def plot_multi_bar_chart(data, title, legend_labels, save=None):
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
    bar_width = 0.1

    for i, lbl in enumerate(legend_labels):
        plt.bar(index + i * bar_width, plot_data[i], bar_width, color=color_palette[i], label=lbl)

    plt.xlabel('Gold label')
    plt.ylabel('# samples')
    plt.title(title)
    plt.xticks(index +  bar_width, x_labels)
    plt.legend(bbox_to_anchor=(0,1.12,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=2)
    plt.subplots_adjust(top=0.8)
    plt.show()

def plot_correct_incorrect_bar(x_labels, misclassification_dict, title, block=True):
    num_groups = len(x_labels)
    
    # how many classified as label as array for plotting
    amounts = dict()
    for lbl in x_labels:
        amounts[lbl] = [misclassification_dict[l][lbl] for l in x_labels]
    
    # how many of gold label for acc
    amounts_gold = dict()
    for lbl in x_labels:
        amounts_gold[lbl] = sum([misclassification_dict[lbl][l] for l in x_labels])

    # calculate accuracy
    amount_data = [amounts_gold[lbl] for lbl in x_labels]
    correct = [amounts[lbl][i] for i, lbl in enumerate(x_labels)]
    print(title)
    print(amount_data)
    print(correct)
    accuracy = round(sum(correct) / sum(amount_data) * 100, 2)
    lbl_accuracies = [round(correct[i] / amount_data[i] * 100, 2) for i in range(len(x_labels))]

    fig, ax = plt.subplots()
    index = np.arange(num_groups)
    colors = [color_palette[i] for i in index]
    bar_width = .1

    for i, lbl in enumerate(x_labels):
        plt.bar(index + i * bar_width, amounts[lbl], bar_width, color=color_palette[i], label=lbl)

    plt.xlabel('Gold label')
    plt.ylabel('# samples')
    x_labels = [lbl + ' (' + str(lbl_accuracies[i]) + ')' for i, lbl in enumerate(x_labels)]
    plt.xticks(index +  bar_width, x_labels)
    plt.title(title + ' (' + str(accuracy) + ')')
    plt.legend(title='Classified as:', bbox_to_anchor=(0,1.12,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=3)
    plt.subplots_adjust(top=0.8)
    plt.show(block=block)