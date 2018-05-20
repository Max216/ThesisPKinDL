import sys, os, json
sys.path.append('./../')

VISUALIZE = False

if not VISUALIZE:
    from libs import model_tools, compatability, data_handler, embeddingholder, collatebatch, data_tools
from docopt import docopt

import torch
from torch.utils.data import DataLoader
import torch.autograd as autograd

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import colors
import time
import numpy as np

def main():
    args = docopt("""

    Usage:
        eval_repr_final.py store_repr <model_path> <dataset_path> <out_path>
        eval_repr_final.py matrix <file_path>
    """)

    if args['store_repr']:
        store_repr(args['<model_path>'], args['<dataset_path>'], args['<out_path>'], act=True)
    elif args['matrix']:
        plot_matrix(args['<file_path>'])


def store_repr(model_path, data_path, out_path, act=False):
    text_separator = ' ?@? '

    print('Load embeddings')
    embedding_holder = embeddingholder.create_embeddingholder()
    print('Load data')
    dataholder = data_handler.Datahandler(data_path, data_format='snli', include_start_end_token=True, sort=False)
    print('Load model')
    classifier_name, classifier, embedding_holder = model_tools.load(model_path, embedding_holder)
    print('Predict')

    index_to_tag = data_tools.DEFAULT_VALID_LABELS
    dataset = dataholder.get_dataset_including_sents(embedding_holder)
    amount = len(dataset)
    with open(out_path, 'w') as f_out:
        cnt=1
        data_loader = DataLoader(dataset, drop_last=False, batch_size=1, shuffle=False, collate_fn=collatebatch.CollateBatchIncludingSents(embedding_holder.padding()))
        for premise_batch, hyp_batch, lbl_batch, p_sent, h_sent in data_loader:
            prediction, act, representations = classifier(
                autograd.Variable(premise_batch),
                autograd.Variable(hyp_batch),
                output_sent_info = True
            )

            _, predicted_idx = torch.max(prediction.data, dim=1)
            pred_label = index_to_tag[predicted_idx[0]]
            gold_label = index_to_tag[lbl_batch[0]]

            premise_repr = representations[0][0].data.numpy().tolist()
            hyp_repr = representations[1][0].data.numpy().tolist()

            responsible_w_p = [p_sent[0][v] for v in act[0][0].numpy()]

            f_out.write(gold_label + ' ' + pred_label + '\t')
            f_out.write(' '.join([str(v) for v in premise_repr]) + '\t')
            f_out.write(' '.join([str(v) for v in hyp_repr]) + '\t')
            f_out.write(text_separator.join(responsible_w_p) + '\n')

            print ("\r Progressing: ", cnt, '/', amount, end="")
            cnt += 1

        print()
        print('Done.')


def plot_matrix(file_path):
    T_ANY = 0
    T_ONLY_CORRECT = 1
    T_ONLY_INCORRECT = 2

    BIN_SIZE = 0.1
    MAX_VAL = 50
    LABEL = 'contradiction'
    TYPE = T_ANY


    print('Read data')
    cnt = 0
    gold_label = None
    pred_label = None
    p_repr = None
    h_repr = None
    samples = []
    with open(file_path) as f_in:
        for line in f_in:
            line = line.strip()
            if cnt % 3 == 0:
                labels = line.split(' ')
                gold_label = labels[0]
                pred_label = labels[1]
            elif cnt % 3 == 1:
                p_repr = [float(v) for v in line.split(' ')]
            else:
                h_repr = [float(v) for v in line.split(' ')]
                samples.append((gold_label, pred_label, p_repr, h_repr))

            cnt += 1

    print('Focus on', LABEL)
    samples = [(gold, pred, h_rep, p_rep) for (gold, pred, h_rep, p_rep) in samples if gold == LABEL]
    if TYPE == T_ONLY_CORRECT:
        samples = [(gold, pred, h_rep, p_rep) for (gold, pred, h_rep, p_rep) in samples if pred == LABEL]
    elif TYPE == T_ONLY_INCORRECT:
        samples = [(gold, pred, h_rep, p_rep) for (gold, pred, h_rep, p_rep) in samples if pred != LABEL]
    print(len(samples))

    print('Create matrix')
    labels_p, labels_h, matrix = create_general_matrix(samples, BIN_SIZE)
    print('plot')
    plt_general_confusion_matrix(matrix, labels_h, labels_p, MAX_VAL)

def create_general_matrix(samples, bin_size):
    min_h = min([min(h_rep) for _,_2,_3, h_rep in samples])
    max_h = max([max(h_rep) for _,_2,_3, h_rep in samples])
    min_p = min([min(p_rep) for _,_2, p_rep, _3 in samples])
    max_p = max([max(p_rep) for _,_2, p_rep, _3 in samples])
    print('h', min_h, max_h)
    print('p', min_p, max_p)

    bins_p = int(np.abs(max_p - min_p) // bin_size)
    bins_h = int(np.abs(max_h - min_h) // bin_size)

    labels_p = np.arange(min_p, max_p + 1, bin_size)[:bins_p]
    labels_h = np.arange(min_h, max_h + 1, bin_size)[:bins_h]

    labels_p = [round(v, 3) for v in labels_p]
    labels_h = [round(v, 3) for v in labels_h]

    def get_idx(val_p, val_h, labels_p, labels_h):
        idx_p = 0
        idx_h = 0
        for i in range(len(labels_p)):
            if labels_p[i] > val_p:
                break
            else:
                idx_p = i

        for i in range(len(labels_h)):
            if labels_h[i] > val_h:
                break
            else:
                idx_h = i
        return (idx_p, idx_h)

    matrix = np.zeros((bins_p, bins_h))

    for gold, pred, p_rep, h_rep in samples:
        for i in range(len(p_rep)):
            idx_p, idx_h = get_idx(p_rep[i], h_rep[i], labels_p, labels_h)
            matrix[idx_p, idx_h] += 1

    for idx_p in range(bins_p):
        for idx_h in range(bins_h):
            matrix[idx_p, idx_h] = round(matrix[idx_p, idx_h] / len(samples),1)
    
    return labels_p, labels_h, matrix

def plt_general_confusion_matrix(matrix, label_x, label_y, max_val):
    LBL_NOT_DATA = 5000
    fig, ax = plt.subplots()
    
    label_matrix = np.copy(matrix)

    if max_val != None and len(label_x) > 1 and len(label_y) > 1:
        uncolored_matrix = np.copy(matrix)

        for xi in range(len(label_x) - 1):
            for yi in range(len(label_y) - 1):
                # check upper value

                if matrix[yi, xi] > max_val:
                    matrix[yi, xi] = max_val


        cmap = colors.ListedColormap(['white', '#0000ff00'])
        bounds=[LBL_NOT_DATA,-4000, 0]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        



    cax = ax.imshow(matrix, origin='upper')
    #if max_val != None:
#       empty = ax.imshow(uncolored_matrix, origin='upper',cmap=cmap, norm=norm, interpolation='nearest')
    fig.colorbar(cax)
    plt.xticks(np.arange(len(label_x)), label_x, rotation=45)
    plt.yticks(np.arange(len(label_y)), label_y)
    plt.xlabel('hypothesis')
    plt.ylabel('premise')
    ax.xaxis.tick_top()
    ax.set_xlabel('hypothesis') 

    width, height = matrix.shape
    for x in range(width):
        for y in range(height):
            plt.annotate(str(label_matrix[x,y]), size=6, xy=(y, x), horizontalalignment='center', verticalalignment='center')

    name =  str(time.time()) + '.png'
    #plt.savefig('./data/' + name)
    plt.show()
    return name

if __name__ == '__main__':
    main()