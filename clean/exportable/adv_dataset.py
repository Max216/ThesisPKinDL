'''
This file enables access to the folder based generated adversarial samples and provides a uniform way of
evaluation models on this data.

USAGE:

'''
import os, json, collections

def strround(val, digits=3):
    return str(round(val, digits))

def _parse_data_txt(in_path):
    '''
    Parse the content of the root 'data.txt' of the dataset.
    '''
    with open(in_path) as f_in:
        lines = [line.strip().split(' ') for line in f_in.readlines()]

    return [(line[0], int(line[1]), int(line[2])) for line in lines]

def _parse_group_summary(in_path):
    '''
    Parse the content of the SUMMARY.sjson file.
    '''
    with open(os.path.join(in_path, 'SUMMARY.sjson')) as f_in:
        all_pairs = [json.loads(line.strip()) for line in f_in.readlines()]

    return [(
        pair['word_p'], pair['word_h'], pair['amount'], pair['assumed_label'],
        pair['rel_path'], pair['sents_with_word_p'], pair['sents_with_word_h'],
        pair['real_sample_count'] 
    ) for pair in all_pairs]

def _parse_word_pair(in_path):
    '''
    Parse the content of a wordpair.jsonl file
    '''
    with open(in_path) as f_in:
        all_samples = [json.loads(line.strip()) for line in f_in.readlines()]

    sentences = [(sample['sentence1'], sample['sentence2']) for sample in all_samples]
    replaced = [int(sample['generation_replaced']) for sample in all_samples]

    return (sentences, replaced)

def _print_samples(out_path, word_premise, word_hypothesis, assumed_label, sentences, replacements, predictions, amount):
    '''
    Print sample sentences, sorted by prediction
    '''
    with open(out_path, 'a') as f_out:
        f_out.write('# ' + word_premise + ' - ' + word_hypothesis + ': ' + assumed_label + '\n')
        enumerated_predictions = [(i, prediction) for i, prediction in enumerate(predictions)]
        for label in ['entailment', 'neutral', 'contradiction']:
            # find samples
            indizes = [i for i, prediction in enumerated_predictions if prediction == label][:amount]
            if len(indizes) > 0:
                f_out.write('## predicted as: ' + label + '\n')
                for i in indizes:
                    generated_p = ''
                    generated_h = ''
                    if replacements[i] == 1:
                        generated_h = ' generated'
                    elif replacements[i] == 2:
                        generated_p = ' generated'
                    else:
                        1/0
                    f_out.write('[p' + generated_p + '] ' + sentences[i][0] + '\n')
                    f_out.write('[h' + generated_h + '] ' + sentences[i][1] + '\n\n')



def _filter_word_pairs(word_pairs, min_cnt_word, max_cnt_word_single, max_cnt_word_both, max_real_samples, min_generated_samples):

    def pass_min_w_cnt(cnt1, cnt2):
        if min_cnt_word != None:
            if cnt1 < min_cnt_word or cnt2 < min_cnt_word:
                return False
        return True

    def pass_max_w_cnt_single(cnt1, cnt2):
        if max_cnt_word_single != None:
            if cnt1 > max_cnt_word_single or cnt2 > max_cnt_word_single:
                return False
        return True

    def pass_max_w_cnt_both(cnt1, cnt2):
        if max_cnt_word_both != None:
            if cnt1 > max_cnt_word_both and cnt2 > max_cnt_word_both:
                return False
        return True

    def pass_max_real_samples(cnt):
        if max_real_samples != None:
            if cnt > max_real_samples:
                return False
        return True

    def pass_min_generated_samples(cnt):
        if min_generated_samples != None:
            if cnt < min_generated_samples:
                return False
        return True

    return [
        (w1, w2, num_samples, lbl, relpath, cnt_w1, cnt_w2, cnt_real_samples)
        for w1, w2, num_samples, lbl, relpath, cnt_w1, cnt_w2, cnt_real_samples in word_pairs
        if pass_min_w_cnt(cnt_w1, cnt_w2) and pass_max_w_cnt_single(cnt_w1, cnt_w2) and pass_max_w_cnt_both(cnt_w1, cnt_w2) and pass_max_real_samples(cnt_real_samples) and pass_min_generated_samples(num_samples)
    ]


def evaluate(prediction_fn, dataset_path, output_path, print_samples=None, min_cnt_word=None, max_cnt_word_single=None, max_cnt_word_both=None, max_real_samples=None, min_generated_samples=None):
    '''
    Evaluate a model on the generated dataset. This will output the model's performance over all data and also over every
    specific replacement that has been done to create the adversarial samples.

    :param prediction_fn            Function to predict the label of samples. Input sentences are not tokenized or
                                    preprocessed in any way.
                                    input: list of samples: [(premise, hypothesis), ... ]
                                    output: list of predicted labels (strings): ['predicted_label_1', ... ]
    :param dataset_path             Path to the 'data.txt' of the generated dataset (in the root folder)
    :param output_path              Path to the folder that will be used to store the results of the evaluation
    :print_samples                  Set to the amount of samples that should be stored per word-pair per predicted label
                                    (default: None)
    :param min_cnt_word             Filters out all word pairs if not both of them have been in at least <min_cnt_word> sentences in the train data
                                    (default:None)
    :param max_cnt_word_single      Filters out all word pairs if at least one of the words have been in more than <max_cnt_word_single> sentences
                                    in the train data. 
                                    (default: None)
    :param max_cnt_word_both        Filter out all word pairs if both of the words have been in more than <max_cnt_word_both> sentences in the train data.
                                    (default: None)
    :param max_real_samples         Filters out all word pairs if there have been more than <max_real_samples> samples in the training set containing
                                    the first word within the premise, the 2nd word within the hypothesis and the same label as specified in the word pair.
                                    (default: None)
    :param min_generated_samples    Filters out all word pairs if not at least <min_generated_samples> samples could be generated.
                                    (default: None)
    '''

    dataset_groups = _parse_data_txt(dataset_path)
    dataset_base_folder = os.path.dirname(os.path.realpath(dataset_path))
    output_path = os.path.realpath(output_path)

    # for overall evaluation
    all_word_pair_accuracies = []
    all_word_pair_correct = 0
    all_word_pair_total = 0

    group_accuracies = []
    reweighted_group_accuracies = []

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for group_name, amount_group_pairs, amount_group_samples in dataset_groups:
        word_pairs_folder = os.path.join(dataset_base_folder, group_name)
        word_pairs = _parse_group_summary(word_pairs_folder)
        word_pairs = _filter_word_pairs(word_pairs, min_cnt_word, max_cnt_word_single, max_cnt_word_both, max_real_samples, min_generated_samples)
        
        word_pair_results = []
        sample_output_path = os.path.join(output_path, group_name + '.txt')

        group_total = 0
        group_correct = 0
        accuracies = []
        print('evaluate group:', group_name)
        for word_p, word_h, sample_amount, gold_label, word_pair_rel_path, word_p_cnt, word_h_cnt, real_sample_cnt in word_pairs:
            sentences, replacements = _parse_word_pair(os.path.join(word_pairs_folder, word_pair_rel_path))

            # Evaluate
            prediction_dict = collections.defaultdict(int)
            predicted_labels = prediction_fn(sentences)
            for predicted in predicted_labels:
                prediction_dict[predicted] += 1

            correct = prediction_dict[gold_label]  
            acc = correct / len(predicted_labels)
            accuracies.append(acc)
            print(word_p, '-', word_h, '->', acc)
            
            # update absolute numbers of prediction within the group
            group_correct += correct
            group_total += len(predicted_labels)

            # Update evaluation for group
            word_pair_results.append((correct, len(predicted_labels), prediction_dict))

            if print_samples != None:
                _print_samples(sample_output_path, word_p, word_h, gold_label, sentences, replacements, predicted_labels, print_samples)

        # Update evaluation for total
        all_word_pair_accuracies.extend(accuracies)

        all_corrects, all_totals, all_pred_dicts = map(list, zip(*word_pair_results))
        all_word_pair_correct += sum(all_corrects)
        all_word_pair_total += sum(all_totals)
        group_accuracies.append(group_correct / group_total)
        reweighted_group_accuracies.append(sum(accuracies) / len(accuracies))

        # write out result csv for this group
        csv_output_path = os.path.join(output_path, group_name + '.csv')
        with open(csv_output_path, 'w') as f_out:
            # print header
            f_out.write('premise word,hypothesis word,label,recall/acc,# samples,# pred. entailment,# pred. neutral,# pred. contradiction,# sents w/ w1,# sent w/ w2,# real samples,comment\n')
            
            # print results
            for i in range(len(word_pairs)):
                word_p, word_h, sample_amount, gold_label, word_pair_rel_path, word_p_cnt, word_h_cnt, real_sample_cnt = word_pairs[i]
                line = [
                    word_p, word_h, gold_label,
                    strround(all_corrects[i] / all_totals[i]), str(all_totals[i]),
                    str(all_pred_dicts[i]['entailment']), str(all_pred_dicts[i]['neutral']), str(all_pred_dicts[i]['contradiction']),
                    str(word_p_cnt), str(word_h_cnt), str(real_sample_cnt), '-'
                ]
                f_out.write(','.join(line) + '\n')

    # write overall result
    csv_summary_path = os.path.join(output_path, 'SUMMARY.csv')
    with open(csv_summary_path, 'w') as f_out:
        # print header
        f_out.write('group,acc,re-weighted acc,# wordpairs,#samples\n')

        # print out results
        for i, (group_name, amount_group_pairs, amount_group_samples) in enumerate(dataset_groups):
            line = [
                group_name, 
                strround(group_accuracies[i]), strround(reweighted_group_accuracies[i]),
                str(amount_group_pairs), str(amount_group_samples)
            ]

            f_out.write(','.join(line) + '\n')

    overall_accuracy = all_word_pair_correct / all_word_pair_total
    overall_reweighted_accuracy = sum(all_word_pair_accuracies) / len(all_word_pair_accuracies)


    print('DONE.')
    print('Overall accuracy:', strround(overall_accuracy, 4))
    print('Reweighted accuracy:', strround(overall_reweighted_accuracy, 4))

                




