import sys
import os
import matplotlib.pyplot as plt
import torch
import torch.autograd as autograd
import torch.nn.functional as F
from torch.utils.data import DataLoader
import model
from model import cuda_wrap, EntailmentClassifier, load_model_state

from docopt import docopt
import re

import nltk
from nltk import word_tokenize

import embeddingholder
import config
import mydataloader
import train

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches


FOLDER_ANALYSES = './analyses/'

class OutWriter:
    def __init__(self, premise, hypothesis, dir, console_print=False):
        self.name = '_'.join(premise) + '_'.join(hypothesis) + 'analysis.txt'
        self.console_print = console_print
        self.dir = dir
        self.txt = []

    def h1(self, h1):
        self.put()
        self.put('#' * len(h1))
        self.put(h1)
        self.put('#' * len(h1))
        self.put()

    def h2(self, h2):
        self.put()
        self.put(h2)
        self.put('=' * len(h2))

    def h3(self, h3):
        self.put('§ ' + h3)

    def put(self, line=''):
        self.txt.append(line + '\n')

        if self.console_print:
            print(line)

    def put_all(self, lines):
        for line in lines:
            self.put(line)

    def finalize(self):
        path = self.dir + self.name
        print('Writing output into', path, '...')
        with open(path, 'w') as f_out:
            for line in self.txt:
                f_out.write(line)
        print('Done.')



def left_number(val):
    '''
    Remove the letters from a value of a model name. e.g. 0_001lr -> 0.001
    '''
    return re.split('[a-z]', val)[0]

def lbl_to_float(val):
    '''
    Map a value from a name to a float.
    '''
    return float(val.replace('_', '.'))

def idx_to_w(idxs, word_dict):
    return [word_dict[i] for i in idxs.data.numpy()]


def analyse(model, data, embedding_holder):
    '''
    Analyse what the model learned by checking where max-activation from the sentence
    representation are coming from.
    '''

    loader = DataLoader(data, 
                        drop_last = False,   
                        batch_size=1, 
                        shuffle=False, 
                        #num_workers=0, 
                        collate_fn=train.CollocateBatch(embedding_holder.padding()))


    for i_batch, (batch_p, batch_h, batch_lbl) in enumerate(loader):
        premises = autograd.Variable(cuda_wrap(batch_p))
        hypothesis = autograd.Variable(cuda_wrap(batch_h))
        predictions, indizes, sent_representations = model(premises, hypothesis, output_sent_info=True)

        predictions = predictions.data

        indizes_p = indizes[0].data
        indizes_h = indizes[1].data
        sent_repr_p = sent_representations[0].data.squeeze().numpy()
        sent_repr_h = sent_representations[1].data.squeeze().numpy()
        last_index_forward = indizes[0].size()[1] // 2 - 1

        # Can return here, has only one value
        return (indizes_p, indizes_h, sent_repr_p, sent_repr_h, last_index_forward, predictions)


def examine_sent(raw, indizes):
    # no need to make it efficient since only very low scale
    result = []
    for i_word, word in enumerate(raw):
        word_activations = [position for position, val in enumerate(indizes) if val == i_word]
        result.append(word_activations)

    return result

def plot_word_activations(orig_words, words, activations, last_idx_forward, directory):

    # add information about the amount of activations per word
    absolute_amounts = [len(a) for a in activations]
    num_all = sum(absolute_amounts)
    relative_amounts = [round(abs_amount / num_all, 2) for abs_amount in absolute_amounts]
    word_labels = [words[i] + ' \n(' + str(absolute_amounts[i]) + ')(' + str(relative_amounts[i]) + ')' for i in range(len(words))]

    y_axis = [i for i in reversed(range(len(words)))]
    colors = [cm.Accent(float(i) / len(words)) for i in y_axis]

    plt.yticks(y_axis, word_labels)
    line_x = last_idx_forward + 0.5
    plt.plot((line_x, line_x), (0, y_axis[0]), 'k-', linewidth=0.2)
    for i in range(len(y_axis)):
        y_value = [y_axis[i] for idx in range(len(activations[i]))]
        plt.scatter(activations[i], y_value, s=1, c=colors[y_axis[i]])
    plt.title(' '.join(orig_words))
    plt.tight_layout()

    path = directory + '_'.join(orig_words) +'analyse-words.png'
    plt.savefig(path)
    plt.clf()

    return path

def plot_sentence_representation(orig_words, words, repr, indizes, directory, min_val, max_val):
    '''
    plot the sentence representations as a bar chart
    @param orig_words   as they appear in the sentence
    @param words        with additional information: UNK
    @param repr         sentence representation
    @param indizes      word indizes with max value
    @param directory    to save the file into
    @param min_val      lowest value for plotting
    @max_val            highest value for plotting
    '''



    colors = [cm.Accent(float(i) / len(words)) for i in range(len(words))]
    x = range(len(repr))
    plt.title(' '.join(orig_words))
    barlist = plt.bar(x, repr)

    # colorize
    for i in range(len(barlist)):
        barlist[i].set_color(colors[indizes[i]])

    # legend
    patches = [mpatches.Patch(color=colors[i], label=words[i]) for i in range(len(colors))]
    lgd = plt.legend(handles=patches, bbox_to_anchor=(1,-.07))
    #plt.gcf().subplots_adjust(bottom=0.07 * len(words))


    plt.axis([0,len(repr), min_val,max_val])
    path = directory + '_'.join(orig_words) +'analyse-representation.png'
    plt.savefig(path, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.clf()

    return path

def concrete_word_activations(words, max_indizes, sent_repr):
    '''
    Assign all activation indizes to each corresponding word.

    @return a list of strings, each string per word.
    '''
    w_with_indizes = [(words[i], max_indizes[i]) for i in range(len(words))]

    # create string
    stringed_words = []
    for i in range(len(w_with_indizes)):
        w, indizes = w_with_indizes[i]
        w = w + '<' + str(i) + '>:\n'
        indizes = [str(idx) + '=' + str(round(sent_repr[idx],3)) for idx in indizes]
        stringed_words.append(w + '; '.join(indizes) + '\n')

    return stringed_words


def find_max_min_overlap(activations_p, activations_h, representation_p, representation_h, amount=5):
    '''
    Find the maximum/minimum overlap of the given activations between hypothesis and premise

    @param activations_p    activation indizes of the premise
    @param activations_h    activation indizes of the hypothesis
    '''

    all_tuples = [(ip, ih) for ih in range(len(activations_h)) for ip in range(len(activations_p))]


    # most amount of same activations in total
    overlap_absolute = [(ip, ih, np.intersect1d(activations_p[ip], activations_h[ih])) for ip, ih in all_tuples]

    # Add information about relative overlap:
    # > (idx_premise, idx_hypothesis, absolute overlap, rel overlap of p, rel overlap of h)

    # how much percent of the premise activations is covered in the hypothesis word
    overlap_absolute_relative = [(ip, ih, intersect, len(intersect) / len(activations_p[ip]), len(intersect) / len(activations_h[ih])) for ip, ih, intersect in overlap_absolute]

    # add values
    activation_values = [(
        [representation_p[intersect[j]] for j in range(len(intersect))],
        [representation_h[intersect[j]] for j in range(len(intersect))]) 
    for ip,ih,intersect,_,__ in overlap_absolute_relative]

    # add info: 
    # - sum of intersection activations premise
    # - sum of intersection activations hypothesis
    # - intersection values absolute difference
    # - avg intersection values absolute difference

    def avg(value, divisor):
        if divisor != 0:
            return value/divisor
        else:
            return '-'

    overlap_abs_rel_values = [(ip, ih, intersect, rel_p, rel_h, 
        sum(activation_values[i][0]),
        sum(activation_values[i][1]),
        sum(np.absolute(np.asarray(activation_values[i][0]) - np.asarray(activation_values[i][1]))),
        avg(sum(np.absolute(np.asarray(activation_values[i][0]) - np.asarray(activation_values[i][1]))),len(intersect))
        ) for i, (ip, ih, intersect, rel_p, rel_h) in enumerate(overlap_absolute_relative)]

    # get max/min overlaps
    sorted_absolute = sorted(overlap_abs_rel_values, key=lambda x: len(x[2]), reverse=True)
    most_overlap_absolute = sorted_absolute[:amount]
    least_overlap_absolute = sorted_absolute[-amount:]
    
    sorted_relative_p_to_h = sorted(overlap_abs_rel_values, key=lambda x: x[3], reverse=True)
    most_overlap_relative_p_to_h = sorted_relative_p_to_h[:amount]
    least_overlap_relative_p_to_h = sorted_relative_p_to_h[-amount:]

    sorted_relative_h_to_p = sorted(overlap_abs_rel_values, key=lambda x: x[4], reverse=True)
    most_overlap_relative_h_to_p = sorted_relative_h_to_p[:amount]
    least_overlap_relative_h_to_p = sorted_relative_h_to_p[-amount:]

    avg_value_difference = sorted([x for x in overlap_abs_rel_values if x[-1] is not '-'], key=lambda x: x[-1], reverse=True)
    most_abs_value_diff = avg_value_difference[:amount]
    least_abs_value_diff = avg_value_difference[-amount:]

    return (
        most_overlap_absolute, least_overlap_absolute, 
        most_overlap_relative_p_to_h, least_overlap_relative_p_to_h,
        most_overlap_relative_h_to_p, least_overlap_relative_h_to_p,
        most_abs_value_diff, least_abs_value_diff
        )

def analyse_single_sent(words, activations, representation):
    analysed_sent = [(
        words[i], 
        len(activations[i]), 
        sum([representation[activations[i][j]] for j in range(len(activations[i]))]),
        sum(np.absolute(np.asarray([representation[activations[i][j]] for j in range(len(activations[i]))]))),
        min([representation[activations[i][j]] for j in range(len(activations[i]))]),
        max([representation[activations[i][j]] for j in range(len(activations[i]))]),
        [(activations[i][j], representation[activations[i][j]]) for j in range(len(activations[i]))]
        ) for i in range(len(words))]

    base_stats =  [word + ' (activations: ' + str(amount_act) + \
                ', sum: ' + str(round(sum_act, 3)) + ', absolute sum: ' + str(round(abs_sum_act, 3)) + \
                ', minimum: ' + str(round(min_act, 3)) + ', maximum: ' + str(round(max_act, 3))  + ')'
        for word ,amount_act, sum_act, abs_sum_act, min_act, max_act, _ in analysed_sent]

    def activations_to_str(val):
        activations = val[-1]
        return '; '.join([str(idx) + '=' + str(round(act, 3)) for idx, act in activations])

    all_activations = [activations_to_str(v) for v in analysed_sent]

    return base_stats, all_activations

def analyse_overlap(premise, hypothesis, activations_p, activations_h, representation_p, representation_h, amount=5):
    # Todo
    # iterate via 1-1 , 1-2, 2-1, 2-2 0-1 1-0
    # stringify results
    unigram_overlap = find_max_min_overlap(activations_p, activations_h, representation_p, representation_h, amount=amount)
    return [unigram_overlap]

def repr_idx_to_w_idx(repr_idx, indizes):
    return indizes[repr_idx]

def find_similarities_unsimilarities(premise, hypothesis, indizes_p, indizes_h, representation_p, representation_h, amount):
    differences = sorted([(i,x) for i,x in enumerate(np.absolute(representation_p - representation_h))], key=lambda x: x[1])
    
    # find min max and add word info
    max_diff = [(idx, diff, repr_idx_to_w_idx(idx, indizes_p), repr_idx_to_w_idx(idx, indizes_h)) for idx, diff in differences[-amount:]][::-1] # reverse
    min_diff = [(idx, diff, repr_idx_to_w_idx(idx, indizes_p), repr_idx_to_w_idx(idx, indizes_h)) for idx, diff in differences[:amount]]

    # to output format
    def to_format(idx_repr, diff, w_idx_p, w_idx_h):
        return '[' + str(idx_repr) + '] Difference: ' + str(round(diff, 3)) + ': \t' + \
            str(premise[w_idx_p]) + ' (premise[' + str(w_idx_p) + ']) \t' +\
            str(hypothesis[w_idx_h]) + ' (hypothesis[' + str(w_idx_h) + '])'

    def to_out(result):
        return [to_format(idx, diff, w_idx_p, w_idx_h) for idx, diff, w_idx_p, w_idx_h in result]

    
    return (to_out(max_diff), to_out(min_diff))


def stringify_overlap(premise, hypothesis, overlap):

    def round_with_str(val, digits):
        if isinstance(val, float):
            return round(val, digits)
        else:
            return val

    return [premise[ip] + ' (premise[' + str(ip) + ']) \t ' + hypothesis[ih] + ' (hypothesis[' + str(ih) + '])\n' + \
        'intersect_len=' + str(len(intersect)) + ', percent_premise=' + str(round(percent_p, 2)) + ', percent_hyp=' + str(round(percent_h, 2)) + '\n' + \
        'intersect_sum_premise=' + str(round(sum_repr_p, 3)) + ', intersect_sum_hyp=' + str(round(sum_repr_h)) + '\n' +\
        'intersect_abs_diff=' + str(round(sum_abs_diff, 3)) + ', intersect_avg_abs_diff=' + str(round_with_str(avg_abs_diff, 3)) + '\n' +\
        'intersect=' + ','.join([str(i) for i in intersect])
        for ip, ih, intersect, percent_p, percent_h, sum_repr_p, sum_repr_h, sum_abs_diff, avg_abs_diff in overlap
    ]

def load_model(model_path, embedding_holder):
    '''
    Loads a trained model. If not renamed, parameters can be regained from the model's name.
    '''
    model_name = model_path.split('/')[-1]
    splitted = model_name.split('-')

    lr = lbl_to_float(left_number(splitted[0]))
    hidden_dim = int(left_number(splitted[1]))
    lstm_dim = [int(i) for i in left_number(splitted[2]).split('_')]
    batch_size = int(left_number(splitted[3]))
    dropout = lbl_to_float(left_number(splitted[6]))

    if splitted[5] == 'relu':
        nonlinearity = F.relu
    else:
        raise Eception('Unknown activation function.', splitted[5])


    
    model = cuda_wrap(EntailmentClassifier(embedding_holder.embeddings, 
                                            dimen_hidden=hidden_dim, 
                                            dimen_out=3, 
                                            dimen_sent_encoder=lstm_dim,
                                            nonlinearity=nonlinearity, 
                                            dropout=dropout))

    print('Load model ...')
    model.load_state_dict(load_model_state(model_path))
    model.eval()
    print('Loaded.')

    return model, model_name

def run(model, model_name, embedding_holder, premise, hypothesis, amount=5):
    '''
    Use this method to automatically trigger the analysis process from within a program.
    '''

    dummy_label = 'neutral' # is not used anyway
    data = mydataloader.SNLIDataset([(premise, hypothesis, dummy_label)], embedding_holder)

    idx_p, idx_h, repr_p, repr_h, last_idx_forward, predictions = analyse(model, data, embedding_holder)

    # create destination for results
    directory = FOLDER_ANALYSES + model_name + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Analyze prediction
    predictions = predictions.squeeze().numpy()
    idx_p = idx_p.squeeze().numpy()
    idx_h = idx_h.squeeze().numpy()
    predictions = ' - '.join([str(predictions[i]) + '[' + mydataloader.index_to_tag[i] + ']' for i in range(len(predictions))])

    premise_original = premise[:]
    hypothesis_original = hypothesis[:]

    premise = embedding_holder.replace_unk(premise)
    hypothesis = embedding_holder.replace_unk(hypothesis)

    result_p = examine_sent(premise, idx_p)
    result_h = examine_sent(hypothesis, idx_h)


    # SINGLE SENT ANALYSIS

    ow = OutWriter(premise_original, hypothesis_original, directory, console_print=True)
    ow.h1('Analysis')
    ow.h3('premise:')
    ow.put(' '.join(premise_original))
    ow.h3('hypothesis:')
    ow.put(' '.join(hypothesis_original))
    ow.h3('predictions:')
    ow.put(predictions)
    ow.h2('Overview Plots')
    ow.put('Look at the .png files:')
    ow.put()


    # Plot overview
    plt_act_p = plot_word_activations(premise_original, premise, result_p, last_idx_forward, directory)
    plt_act_h = plot_word_activations(hypothesis_original, hypothesis, result_h, last_idx_forward, directory)

    min_val = np.min([np.min(repr_p), np.min(repr_h)])
    max_val = np.max([np.max(repr_p), np.max(repr_h)])
    plt_repr_p = plot_sentence_representation(premise_original, premise, repr_p, idx_p, directory, min_val, max_val)
    plt_repr_h = plot_sentence_representation(hypothesis_original, hypothesis, repr_h, idx_h, directory, min_val, max_val)

    out_p_activations = concrete_word_activations(premise, result_p, repr_p)
    out_h_activations = concrete_word_activations(hypothesis, result_h, repr_h)

    ow.h3('Premise')
    ow.put('activation indizes:')
    ow.put(plt_act_p)
    ow.put('activation values:')
    ow.put(plt_repr_p)
    ow.put()
    ow.h3('Hypothesis')
    ow.put('activation indizes:')
    ow.put(plt_act_h)
    ow.put('activation values:')
    ow.put(plt_repr_h)

    ow.h1('Part of single words in the sentence representation')
    ow.h2('Premise')
    analysed_p_general, analysed_p_act = analyse_single_sent(premise, result_p, repr_p)
    for i in range(len(analysed_p_general)):
        ow.h3(analysed_p_general[i])
        ow.put(analysed_p_act[i])
        ow.put()

    ow.h2('Hypothesis')
    analysed_h_general, analysed_h_act = analyse_single_sent(hypothesis, result_h, repr_h)  
    for i in range(len(analysed_h_general)):
        ow.h3(analysed_h_general[i])
        ow.put(analysed_h_act[i])
        ow.put()

    # BOTH SENT ANALYSIS
    ow.h1('Similarities of sentence representation')
    # sentence representation comparison
    amount_similarities = 50
    unsimilarities, similarites = find_similarities_unsimilarities(premise, hypothesis, idx_p, idx_h, repr_p, repr_h, amount=amount_similarities)
    ow.h2('Similarities')
    ow.put_all(similarites)
    ow.h2('Unsimilarities')
    ow.put_all(unsimilarities)

    # overlap 
    overlaps = analyse_overlap(premise, hypothesis, result_p, result_h, repr_p, repr_h, amount=amount)

    ow.h1('Comparison of unigrams in premise and hypothesis')
    ow.put('Compare which words in premise and hypothesis have similar influence on the representation.')
    ow.h3('intersect_len')
    ow.put('How many same indizes in the sentence representations are coming from these words.\n')
    ow.h3('percent_premise')
    ow.put('How many (in %) indizes in the sentence representation that come from the premise word are covered by the ones from the hypothesis word.\n')
    ow.h3('percent_hyp')
    ow.put('As above, vice versa.\n')
    ow.h3('intersect_sum_premise')
    ow.put('Sum of all values in the premise representation of the shared indizes between premise and hypothesis (not absolute).\n')
    ow.h3('intersect_sum_hyp')
    ow.put('Same as above, for the hypothesis representation.\n')
    ow.h3('intersect_avg_abs_diff')
    ow.put('Averaged absolute difference between the values of both representations with shared indizes.\n')
    ow.h3('intersect')
    ow.put('The actual indizes that both words have contributed to their sentence representation.\n')

    (most_overlap_absolute, least_overlap_absolute, 
    most_overlap_relative_p_to_h, least_overlap_relative_p_to_h,
    most_overlap_relative_h_to_p, least_overlap_relative_h_to_p,
    most_abs_value_diff, least_abs_value_diff) = overlaps[0]

    ow.h2('Pairs with most same indizes in representation [intersect_len]')
    for line in stringify_overlap(premise, hypothesis, most_overlap_absolute):
        ow.put(line)
        ow.put()

    ow.h2('Pairs with least same indizes in representation [intersect_len]')
    for line in stringify_overlap(premise, hypothesis, least_overlap_absolute):
        ow.put(line)
        ow.put()

    ow.h2('Pairs with most index coverage of the premise word [percent_premise]')
    for line in stringify_overlap(premise, hypothesis, most_overlap_relative_p_to_h):
        ow.put(line)
        ow.put()

    ow.h2('Pairs with least index coverage of the premise word [percent_premise]')
    for line in stringify_overlap(premise, hypothesis, least_overlap_relative_p_to_h):
        ow.put(line)
        ow.put()

    ow.h2('Pairs with most index coverage of the hypothesis word [percent_hyp]')
    for line in stringify_overlap(premise, hypothesis, most_overlap_relative_h_to_p):
        ow.put(line)
        ow.put()

    ow.h2('Pairs with least index coverage of the hypothesis word [percent_hyp]')
    for line in stringify_overlap(premise, hypothesis, least_overlap_relative_h_to_p):
        ow.put(line)
        ow.put()

    ow.h2('Pairs with least average elementwise difference of intersecting indizes [intersect_avg_abs_diff]')
    for line in stringify_overlap(premise, hypothesis, least_abs_value_diff):
        ow.put(line)
        ow.put()

    ow.h2('Pairs with most average elementwise difference of intersecting indizes [intersect_avg_abs_diff]')
    for line in stringify_overlap(premise, hypothesis, most_abs_value_diff):
        ow.put(line)
        ow.put()
    # Finalze file
    ow.finalize()

def get_embedding_holder():
    return embeddingholder.EmbeddingHolder(config.PATH_WORD_EMBEDDINGS)

def main():
    args = docopt("""Analyse the model.

    Usage:
        analyse.py <model> <premise> <hypothesis> 

        <model>         Path to trained model that gets analysed.
        <premise>       Premise as a sentence.
        <hypothesis>    Hypothesis as a sentence
    """)

    model_path = args['<model>']
    premise = word_tokenize(args['<premise>'])
    hypothesis = word_tokenize(args['<hypothesis>'])
    
    embedding_holder = get_embedding_holder()
    model, model_name = load_model(model_path, embedding_holder)
    run(model, model_name, embedding_holder, premise, hypothesis)
    

if __name__ == '__main__':
    main()


