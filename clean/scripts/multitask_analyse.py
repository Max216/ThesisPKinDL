from docopt import docopt
import sys, os
sys.path.append('./../')

from libs import multitask, embeddingholder, data_handler

import collections

def main():
    args = docopt("""Analyse multitask data.

    Usage:
        multitask_analyse.py freq_avg <data_res>
    """)

    if args['freq1']:
        calc_multitask_average_per_sent(args['<data_res>'])


def calc_multitask_average_per_sent(res_path):
    datahandler_train = data_handler.get_datahandler_train()
    embedding_holder = embeddingholder.create_embeddingholder()

    train_set, next_id = datahandler_train.get_dataset_id(embedding_holder, start_id=0)
    mt = MultiTaskTarget([train_set], res_path, embedding_holder)

    words, labels, has_content = mt.get_targets(make_even_dist=False)

    counter_samples_content = 0
    counter_samples_no_content = 0

    cnt_e = 0
    cnt_c = 0

    tag_to_idx = mt.tag_to_idx
    idx_to_tag = [None for k in tag_to_idx]
    for k in tag_to_idx:
        idx_to_tag[tag_to_idx[k]] = k

    for i in range(len(has_content)):
        if has_content[i]:
            counter_samples_content += 1
            counter = collections.Counter([idx_to_tag[lbl] for lbl in labels[i].numpy()])
            cnt_e += counter['entailment']
            cnt_c +=  counter['contradiction']
        else:
            counter_samples_no_content += 1

    print('Results for samples having targets:', counter_samples_content, 'not having targets:', counter_samples_no_content, 'total individual sentences:', counter_samples_no_content + counter_samples_content)
    print('Total targets:', cnt_c + cnt_e)
    print('Entailing:', cnt_e, 'contradicting:', cnt_c)
    print('Avg per individual sentence:', 'entailment:', cnt_e / counter_samples_content, '; contradiction:', cnt_c / counter_samples_content, '; total: ', (cnt_e + cnt_c) / counter_samples_content)
    print('Avg including sentences w/o targets: entailment', cnt_e / (counter_samples_content + counter_samples_no_content), '; contradiction:', cnt_c / (counter_samples_content + counter_samples_no_content), '; total:', (cnt_e + cnt_c) / (counter_samples_content + counter_samples_no_content))



if __name__ == '__main__':
    main()
