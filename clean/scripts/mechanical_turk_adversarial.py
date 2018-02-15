from docopt import docopt

import json, re, collections, codecs, os, random
import csv as csv_lib
import numpy as np
from sklearn.metrics import cohen_kappa_score
import math


def csv(in_path, out_path):
    out_lines = []

    with open(in_path) as f_in:
        parsed = [json.loads(line.strip()) for line in f_in.readlines()]

    premise_dict = collections.defaultdict(lambda: [])
    for p in parsed:
        premise_dict[p['sentence1']].append(p)


    with codecs.open(out_path, 'w', 'utf-8') as f_out:
        f_out.write('premise,hypothesis1,hypothesis2,hypothesis3,hypothesis4,hypothesis5,id1,id2,id3,id4,id5\n')
        for premise in premise_dict:
            current = premise_dict[premise]

            chunks = [current[x:x+5] for x in range(0, len(current), 5)]
            print('chunks', len(chunks))
            for chunk in chunks:
                if len(chunk) == 5:
                    print('in chunk', len(chunk))
                    current_hit = []
                    ids = []
                    premise = None
                    for i, sample in enumerate(chunk):
                        premise = sample['sentence1']
                        hypothesis = sample['sentence2']
                        label = sample['gold_label']
                        category = 'synonyms' #sample['category']
                        #replaced1 = sample['replaced1']
                        #replaced2 = sample['replaced2']

                        replacement =sample["replacement"].split(',')
                        replaced1 = replacement[0]
                        replaced2 = replacement[1]

                        _id = sample['id']

                        rep2_regexp = re.compile('\\b' + replaced2 + '\\b')
                        split_hyp = re.split(rep2_regexp, hypothesis)
                        split_prem = re.split(rep2_regexp, premise)

                        new_hyp = ("<span class='highlight'>" + replaced2 + '</span>').join(split_hyp)
                        current_hit.append(new_hyp.replace(',', '&#44;').replace('\"', '&quot;'))
                        ids.append(str(_id))

                    out = [premise.replace(',', '&#44;').replace('\"', '&quot;')] 
                    #for sample in chunk:
                    new_out = out + current_hit + ids
                    f_out.write(u','.join(new_out) + os.linesep)
    print(len(premise_dict.keys()))


def id(in_path, out_path):
    with open(in_path) as f_in:
        parsed = [json.loads(line.strip()) for line in f_in.readlines()]
    
    id_samples = []
    for i, p in enumerate(parsed):
        p['id'] = i + 20000
        id_samples.append(p)

    with open(out_path, 'w') as f_out:
        for p in id_samples:
            f_out.write(json.dumps(p) + '\n')


def filter_common(in_path, out_path):
    filter_cats = set(['colors', 'at-verbs', 'instruments', 'fruits', 'synonyms', 'fastfood'])
    with open(in_path) as f_in:
        parsed = [json.loads(line.strip()) for line in f_in.readlines()]

    print('Samples before filtering:', len(parsed))
    parsed = [p for p in parsed if p['category'] not in filter_cats]
    print('After filtering', len(parsed))

    sample_dict = collections.defaultdict(lambda : [])
    for p in parsed:
        sample_dict[p['sentence1']].append(p)

    rm_keys = []
    for key in sample_dict:
        if len(sample_dict[key]) < 5:
            rm_keys.append(key)

    print('removee because inufficient:', len(rm_keys))
    for k in rm_keys:
        sample_dict.pop(k, None)

    keep_samples = []
    for k in sample_dict:
        keep_amount = len(sample_dict[k]) // 5
        add = sample_dict[k][:keep_amount*5]
        print('add', len(add), 'to keep')
        keep_samples.extend(add)

    print('final samples:', len(keep_samples))
    with open(out_path, 'w') as f_out:
        for sample in keep_samples:
            f_out.write(json.dumps(sample) + '\n')

def verify(file_in):
    with open(file_in) as f_in:
        csv_reader = csv_lib.reader(f_in)
        content = [row for row in csv_reader]

    header = content[0]
    contents = content[1:]

    #id_headers = ["Input.id1", "Input.id2","Input.id3","Input.id4","Input.id5"]
    question_1_headers = ["Answer.label1_hyp1","Answer.label1_hyp2", "Answer.label1_hyp3","Answer.label1_hyp4","Answer.label1_hyp5"]
    question_2_headers = ["Answer.label2_hyp1", "Answer.label2_hyp2", "Answer.label2_hyp3", "Answer.label2_hyp4", "Answer.label2_hyp5"]
    question_3_headers = ["Answer.nonsense_hyp1","Answer.nonsense_hyp2","Answer.nonsense_hyp3","Answer.nonsense_hyp4","Answer.nonsense_hyp5"]

    #sample_id_indizes = [header.index(s_id) for s_id in id_headers]
    sample_q1_indizes = [header.index(q1) if q1 in header else -1 for q1 in question_1_headers]
    sample_q2_indizes = [header.index(q2) if q2 in header else -1 for q2 in question_2_headers]
    sample_q3_indizes = [header.index(q3) if q3 in header else -1 for q3 in question_3_headers]

    if min(sample_q1_indizes + sample_q2_indizes + sample_q3_indizes) < 0:
        print('invalid')
        print(sample_q1_indizes)
        print(sample_q2_indizes)
        print(sample_q3_indizes)
    else:
        print('valid')



def duplicate(csv_in, csv_out):
    result_dict = collections.defaultdict(lambda: [])
    with open(csv_in) as f_in:
        lines = [line for line in f_in.readlines()]

    cnt = 0
    for line in lines:
        if cnt > 0:
            hitiid = line.split(',')[0]
            if not line.endswith('\n'):
                line += '\n'
            result_dict[hitiid].append(line)
        cnt += 1

    for key in result_dict:
        while len(result_dict[key]) < 3:
            result_dict[key].append(result_dict[key][0])

    with open(csv_out, 'w') as f_out:
        f_out.write(lines[0])
        for key in result_dict:
            for line in result_dict[key]:
                f_out.write(line)

def create_result_json(csv_in, file_out):
    with open(csv_in) as f_in:
        csv_reader = csv_lib.reader(f_in)
        content = [row for row in csv_reader]

    header = content[0]
    print('header', header)
    content = [c for i,c in enumerate(content) if i != 0]

    print(len(content))

    question_1_headers = ["Answer.label1_hyp1","Answer.label1_hyp2", "Answer.label1_hyp3","Answer.label1_hyp4","Answer.label1_hyp5"]
    question_2_headers = ["Answer.label2_hyp1", "Answer.label2_hyp2", "Answer.label2_hyp3", "Answer.label2_hyp4", "Answer.label2_hyp5"]
    question_3_headers = ["Answer.nonsense_hyp1","Answer.nonsense_hyp2","Answer.nonsense_hyp3","Answer.nonsense_hyp4","Answer.nonsense_hyp5"]
    id_headers = ["Input.id1","Input.id2","Input.id3","Input.id4","Input.id5"]

    sample_id_indizes = [header.index(s_id) for s_id in id_headers]
    sample_q1_indizes = [header.index(q1) if q1 in header else -1 for q1 in question_1_headers]
    sample_q2_indizes = [header.index(q2) if q2 in header else -1 for q2 in question_2_headers]
    sample_q3_indizes = [header.index(q3) if q3 in header else -1 for q3 in question_3_headers]

    # merge annotations
    result_dict = collections.defaultdict(lambda: [])
    for row in content:
        hit_id = row[0]
        result_dict[hit_id].append(row)

    results = []
    for key in result_dict:
        annotations = result_dict[key]

        samples = []
        for sample_idx in range(len(id_headers)):
            current_sample_labels = []
            for annotation in annotations:
                q1_answer = annotation[sample_q1_indizes[sample_idx]]
                q2_answer = annotation[sample_q2_indizes[sample_idx]]
                q3_answer = annotation[sample_q3_indizes[sample_idx]]
                if q3_answer == 'incorrect':
                    current_sample_labels.append('incorrect')
                elif q1_answer == 'yes' and q2_answer == "":
                    current_sample_labels.append('entailment')
                elif q1_answer == 'yes' and q2_answer == 'add_info':
                    current_sample_labels.append('neutral')
                elif q1_answer == 'no':
                    current_sample_labels.append('contradiction')
                else:
                    1/0
            if len(current_sample_labels) != 3:
                print('no', annotation[sample_id_indizes[sample_idx]])
                1/0

            samples.append((annotations[0][sample_id_indizes[sample_idx]], current_sample_labels))
        if len(samples) != 5:
            1/0

        results.extend(samples)

    with open(file_out, 'w') as f_out:
        for sample_id, labels in results:
            f_out.write(json.dumps({
                'id': sample_id,
                'labels': labels    
            }) + '\n')

def finalize_with_annotations(result_path, src_path, out_path):
    with open(result_path) as f_in:
        results = [json.loads(line.strip()) for line in f_in.readlines()]
        results = [(str(v['id']), v['labels']) for v in results]

    with open(src_path) as f_in:
        data = [json.loads(line.strip()) for line in f_in.readlines()]

    # sort data
    data_dict = dict([(str(d['id']), d) for d in data])
    print(sorted([k for k in data_dict]))

    exclude_categories = set(['movements', 'nationalities', 'countries'])

    processed_amount = 0
    use_data = []
    no_consens = 0
    cnt_bad_cat = 0
    for r_id, r_labels in results:
        # create label
        lbl_cnt = collections.Counter(r_labels)
        best_label, amount = lbl_cnt.most_common()[0]
        sample = data_dict[r_id]
        processed_amount += 1
        if sample['category'] not in exclude_categories:
            sample['annotator_labels'] = r_labels
            if amount >= 2:
                # use
                sample['gold_label'] = best_label
                
            else:
                # not use
                no_consens += 1
                sample['gold_label'] = '-'

            use_data.append(sample)
        else:
            cnt_bad_cat +=1

    print('processed:', processed_amount)
    print('not use because of category:', cnt_bad_cat, 'leaves', processed_amount - cnt_bad_cat, len(use_data))
    print('annotator consense:', len(use_data) - no_consens)
    print('total data inlcuding no consens:', len(use_data))
    print('no consens:', no_consens, 'leaves', len(use_data) - no_consens)

    with open(out_path, 'w') as f_out:
        for d in use_data:
            f_out.write(json.dumps(d) + '\n')

def finalize(result_path, src_path, out_path):
    with open(result_path) as f_in:
        results = [json.loads(line.strip()) for line in f_in.readlines()]
        results = [(str(v['id']), v['labels']) for v in results]

    with open(src_path) as f_in:
        data = [json.loads(line.strip()) for line in f_in.readlines()]

    # sort data
    data_dict = dict([(str(d['id']), d) for d in data])
    print(sorted([k for k in data_dict]))

    labeled_data = []
    unused_data = []
    for r_id, r_labels in results:
        # create label
        lbl_cnt = collections.Counter(r_labels)
        best_label, amount = lbl_cnt.most_common()[0]
        sample = data_dict[r_id]
        if amount >= 2:
            # use
            sample = data_dict[r_id]
            sample['gold_label'] = best_label
            labeled_data.append(sample)
        else:
            # not use
            unused_data.append((sample, r_labels))

    print('annotator consense:', len(labeled_data))
    print('not:', len(unused_data))

    with open(out_path, 'w') as f_out:
        for d in labeled_data:
            f_out.write(json.dumps(d) + '\n')

def stats(new_data_path, old_data_path):
    with open(new_data_path) as f_in:
        new_samples = [json.loads(line.strip()) for line in f_in.readlines()]

    with open(old_data_path) as f_in:
        old_samples = [json.loads(line.strip()) for line in f_in.readlines()]

    old_sample_dict = dict([(p['id'], p) for p in old_samples])

    category_dict = collections.defaultdict(lambda : [])
    for sample in new_samples:
        category_dict[sample['category']].append(sample)

    for cat in category_dict:
        total = len(category_dict[cat])
        incorrect = len([s for s in category_dict[cat] if s['gold_label'] == 'incorrect'])
        new_label = len([s for s in category_dict[cat] if s['gold_label'] != 'incorrect' and s['gold_label'] != old_sample_dict[s['id']]['gold_label']])

        print('#', cat)
        print('Samples:', total)
        print('Incorrect:', incorrect)
        print('different label:', new_label)
        print()

def remove_incorrect(file_in, file_out_correct, file_out_incorrect):
    with open(file_in) as f_in:
        lines = [line for line in f_in.readlines()]

    parsed = [json.loads(line.strip()) for line in lines]

    correct_lines = []
    incorrect_lines = []
    labels = []
    for i, sample in enumerate(parsed):
        label = sample['gold_label']
        if label == 'incorrect':
            incorrect_lines.append(lines[i])
        else:
            correct_lines.append(lines[i])

        labels.append(label)

    print('Labels:', collections.Counter(labels).most_common())

    with open(file_out_correct, 'w') as f_out:
        for line in correct_lines:
            f_out.write(line)

    with open(file_out_incorrect, 'w') as f_out:
        for line in incorrect_lines:
            f_out.write(line)


def analyse(csv_file, data_file, results_file):

    sample_dict = dict()
    results_dict = dict()

    with open(data_file) as f_in:
        parsed = [json.loads(line.strip()) for line in f_in.readlines()]
        for p in parsed:
            sample_dict[str(p['id'])] = '[p] ' + p['sentence1'] + '\n[h] ' + p['sentence2']

    with open(results_file) as f_in:
        parsed = [json.loads(line.strip()) for line in f_in.readlines()]
        for p in parsed:
            results_dict[str(p['id'])] = p['labels']

    LABEL_COMENT = 'Answer.comment'
    ID_HEADERS = ["Input.id1","Input.id2","Input.id3","Input.id4","Input.id5"]
    

    with open(csv_file) as f_in:
        csv_reader = csv_lib.reader(f_in)
        content = [row for row in csv_reader]

    header = content[0]
    content = content[1:]

    comment_idx = header.index(LABEL_COMENT)
    sample_id_indizes = [header.index(s_id) for s_id in ID_HEADERS]

    print('# Comments')
    for c in content:
        cmt = c[comment_idx]
        if cmt != '{}' and len(cmt.strip()) > 0:

            print('## Comment:',c[comment_idx])
            print('## Samples:')
            for i in sample_id_indizes:
                print(sample_dict[str(c[i])])
                print(results_dict[str(c[i])], '\n')

            print()

    print('# No consense')
    for sid in results_dict:
        if len(list(set(results_dict[sid]))) > 2:
            print(sid)
            print(sample_dict[sid])
            print(results_dict[sid])
            print()


# Kappa from https://gist.github.com/skylander86/65c442356377367e27e79ef1fed4adee
def fleiss_kappa(M):
    """
    See `Fleiss' Kappa <https://en.wikipedia.org/wiki/Fleiss%27_kappa>`_.
    :param M: a matrix of shape (:attr:`N`, :attr:`k`) where `N` is the number of subjects and `k` is the number of categories into which assignments are made. `M[i, j]` represent the number of raters who assigned the `i`th subject to the `j`th category.
    :type M: numpy matrix
    """
    N, k = M.shape  # N is # of items, k is # of categories
    n_annotators = float(np.sum(M[0, :]))  # # of annotators

    print('Annotators', n_annotators)

    p = np.sum(M, axis=0) / (N * n_annotators)
    P = (np.sum(M * M, axis=1) - n_annotators) / (n_annotators * (n_annotators - 1))
    Pbar = np.sum(P) / N
    PbarE = np.sum(p * p)

    kappa = (Pbar - PbarE) / (1 - PbarE)

    return kappa


def computeKappa(mat):

    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    DEBUG = True
    """ Computes the Kappa value
        @param n Number of rating per subjects (number of human raters)
        @param mat Matrix[subjects][categories]
        @return The Kappa value """
    n = checkEachLineCount(mat)   # PRE : every line count must be equal to n
    N = len(mat)
    k = len(mat[0])
    
    if DEBUG:
        print(n, "raters.")
        print(N, "subjects.")
        print(k, "categories.")
    
    # Computing p[]
    p = [0.0] * k
    for j in range(k):
        p[j] = 0.0
        for i in range(N):
            p[j] += mat[i][j]
        p[j] /= N*n
    if DEBUG: print("p =", p)
    
    # Computing P[]    
    P = [0.0] * N
    for i in range(N):
        P[i] = 0.0
        for j in range(k):
            P[i] += mat[i][j] * mat[i][j]
        P[i] = (P[i] - n) / (n * (n - 1))
    #if DEBUG: print("P =", P)
    
    # Computing Pbar
    Pbar = sum(P) / N
    #if DEBUG: print("Pbar =", Pbar)
    
    # Computing PbarE
    PbarE = 0.0
    
    # not important, I tried something
    #label_counts = np.sum(mat, axis=0)
    #max_cnt, min_cnt = np.max(label_counts), np.min(label_counts)
    #normalied_counts = softmax(label_counts)

    for i,pj in enumerate(p):
        #if normalize:
        #    factor = normalied_counts[i]
        #else:
        #    factor = 1
        PbarE +=  (pj * pj) #### * factor
    #if DEBUG: print("PbarE =", PbarE)
    
    kappa = (Pbar - PbarE) / (1 - PbarE)
    #if DEBUG: print("kappa =", kappa)
    
    return kappa

def checkEachLineCount(mat):
    """ Assert that each line has a constant number of ratings
        @param mat The matrix checked
        @return The number of ratings
        @throws AssertionError If lines contain different number of ratings """
    n = sum(mat[0])
    
    assert all(sum(line) == n for line in mat[1:]), "Line count != %d (n value)." % n
    return n



def calc_kappa(samples, num_categories = 4):
    print('Calc kappa based on', num_categories, 'categories')
    label_dict = dict([('entailment', 0), ('contradiction', 1), ('neutral', 2), ('incorrect', 3)])
    num_items = len(samples)
    matrix = np.zeros((num_items, num_categories))

    for i,s in enumerate(samples):
        for annotator_label in s['annotator_labels']:
            matrix[i, label_dict[annotator_label]] += 1
    print(matrix)


    
    return computeKappa(matrix)
    #return fleiss_kappa(matrix)


def kappa(path_json, num_labels):
    with open(path_json) as f_in:
        parsed = [json.loads(line.strip()) for line in f_in.readlines()]

    print('# General stats:')
    print('# Labels')
    print(collections.Counter([p['gold_label'] for p in parsed]).most_common())

    # Filtered
    parsed_valid = [p for p in parsed if p['gold_label'] != '-' and p['gold_label'] != 'incorrect']
    print('Valid samples:', len(parsed_valid))

    #print('# Categories')
    #print(collections.Counter([p['category'] for p in parsed_valid]).most_common())

    print('# Fleiss')
    print('size =',len(parsed_valid),'; kappa =', calc_kappa(parsed_valid, num_labels))
    contradiction_samples = [pd for pd in parsed_valid if pd['gold_label'] == 'contradiction']
    entailment_samples = [pd for pd in parsed_valid if pd['gold_label'] == 'entailment']
    neutral_samples = [pd for pd in parsed_valid if pd['gold_label'] == 'neutral']

    print()
    for name, samples in [('contradiction', contradiction_samples), ('entailment',entailment_samples), ('neutral',neutral_samples)]:
        print(name, len(samples))
        print('kappa:', calc_kappa(samples, num_labels))
        print()

    three_agreement = collections.defaultdict(int)
    two_agreement = collections.defaultdict(int)
    any_other_agreement = collections.defaultdict(int)

    for s in parsed_valid:
        if len(list(set(s['annotator_labels']))) == 1:
            three_agreement[s['gold_label']] += 1
        elif len(list(set(s['annotator_labels']))) == 2:
            two_agreement[s['gold_label']] += 1
        else:
            any_other_agreement[s['gold_label']] += 1


    for k in list(set(three_agreement.keys()) | set(two_agreement.keys() | set(any_other_agreement.keys()))):
        total_count = three_agreement[k] + two_agreement[k] + any_other_agreement[k]
        print('Gold label:', k, '; samples:', total_count, 'percentage full agreement:', three_agreement[k] / total_count)
    print('verify if more than two distinct labels are possible. cnt:', sum([any_other_agreement[k] for k in any_other_agreement]))
    print('verify:', sum([three_agreement[k] for k in three_agreement]) + sum([two_agreement[k] for k in two_agreement]) + sum([any_other_agreement[k] for k in any_other_agreement]), '==', len(parsed_valid))
    all_label_dict = collections.defaultdict(lambda: collections.defaultdict(int))
    
    cnt_annotations = 0
    for s in parsed_valid:
        gold = s['gold_label']
        for annotated in s['annotator_labels']:
            all_label_dict[gold][annotated] += 1
            cnt_annotations += 1
    print('annotations total:', cnt_annotations, '->', cnt_annotations/5)

    for gold in all_label_dict:
        cnt_total_annotations = 0
        print('gold label:', gold)
        for annotated in all_label_dict[gold]:
            cnt_total_annotations += all_label_dict[gold][annotated]
        #print('annotations total:', cnt_total_annotations, '->', cnt_total_annotations/5)
        print('verify:', cnt_total_annotations/len(parsed_valid[0]['annotator_labels']), '==', three_agreement[gold] + two_agreement[gold] + any_other_agreement[gold])
        for annotated in all_label_dict[gold]:
            print('labeled as', annotated, ':', all_label_dict[gold][annotated] / cnt_total_annotations)



def create_sets(json_path, out_path):
    with open(json_path) as f_in:
        parsed = [json.loads(line.strip()) for line in f_in.readlines()]

    print(len(parsed), 'samples loaded.')

    # remove incorrect
    out_name = out_path + '.valid_lbls.jsonl'
    print('Write: only keep majority label entailmetn(neutral/contradiction to:', out_name)
    parsed = [p for p in parsed if p['gold_label'] != '-' and p['gold_label'] != 'incorrect']
    print('Write out', len(parsed), 'samples.')
    with open(out_name, 'w') as f_out:
        for p in parsed:
            f_out.write(json.dumps(p) + '\n')
    print('Done.')

    out_name = out_path + '.no_single_incorrect.jsonl'
    print('Write: remove if a single person annotated as incorrect to:', out_name)
    parsed = [p for p in parsed if 'incorrect' not in p['annotator_labels']]
    print('Write out', len(parsed), 'samples.')
    with open(out_name, 'w') as f_out:
        for p in parsed:
            f_out.write(json.dumps(p) + '\n')
    print('Done.')

    out_name = out_path + '.three_agreement.jsonl'
    print('Write: Only keep 3-annotator agreement:', out_name)
    parsed = [p for p in parsed if len(list(set(p['annotator_labels']))) == 1]
    print('Write out', len(parsed), 'samples.')
    with open(out_name, 'w') as f_out:
        for p in parsed:
            f_out.write(json.dumps(p) + '\n')
    print('Done.')


def include_annotator_ids(data_path, csv_path, out_path):
    with open(data_path) as f_in:
        parsed_data = [json.loads(line.strip()) for line in f_in.readlines()]

    with open(csv_path) as f_in:
        csv_reader = csv_lib.reader(f_in)
        content = [row for row in csv_reader]

    header = content[0]
    print('header', header)
    content = [c for i,c in enumerate(content) if i != 0]

    print(len(content))

    question_1_headers = ["Answer.label1_hyp1","Answer.label1_hyp2", "Answer.label1_hyp3","Answer.label1_hyp4","Answer.label1_hyp5"]
    question_2_headers = ["Answer.label2_hyp1", "Answer.label2_hyp2", "Answer.label2_hyp3", "Answer.label2_hyp4", "Answer.label2_hyp5"]
    question_3_headers = ["Answer.nonsense_hyp1","Answer.nonsense_hyp2","Answer.nonsense_hyp3","Answer.nonsense_hyp4","Answer.nonsense_hyp5"]
    id_headers = ["Input.id1","Input.id2","Input.id3","Input.id4","Input.id5"]
    print(header)
    sample_id_indizes = [header.index(s_id) for s_id in id_headers]
    sample_q1_indizes = [header.index(q1) if q1 in header else -1 for q1 in question_1_headers]
    sample_q2_indizes = [header.index(q2) if q2 in header else -1 for q2 in question_2_headers]
    sample_q3_indizes = [header.index(q3) if q3 in header else -1 for q3 in question_3_headers]
    annotator_id = header.index('WorkerId')
    hit_id = header.index('HITId')

    processed_data = []
    for pd in parsed_data:
        pd['annotator_labels'] = []
        pd['annotator_ids'] = []

    print('Load annotations ...')
    cnt = 0
    for pd in parsed_data:
        print('For', cnt)
        cnt += 1
        sample_id = str(pd['id'])
        for i in range(len(content)):
            for id_idx in range(len(sample_id_indizes)):
                id_field = sample_id_indizes[id_idx]
                if str(content[i][id_field]) == sample_id:
                    pd['annotator_ids'].append(content[i][annotator_id])
                    pd['hit_id'] = content[i][hit_id]

                    q1_field = sample_q1_indizes[id_idx]
                    q2_field = sample_q2_indizes[id_idx]
                    q3_field = sample_q3_indizes[id_idx]
                    q1_answer = content[i][q1_field]
                    q2_answer = content[i][q2_field]
                    q3_answer = content[i][q3_field]

                    label = None
                    if q1_answer == 'no':
                        label = 'contradiction'
                    elif q1_answer == 'yes':
                        if q2_answer == '':
                            label = 'entailment'
                        elif q2_answer == 'add_info':
                            label = 'neutral'
                        else:
                            print('unknown q2 answer:', q2_answer)
                            1/0
                    elif q3_answer == 'incorrect':
                        pass
                    else:
                        print('unknown answer:', q1_answer, ';', q2_answer, ';', q3_answer)
                        1/0

                    pd['annotator_labels'].append(label)
    print('Done')
    print('Verify')
    for pd in parsed_data:
        if len(pd['annotator_labels']) != 3:
            print('not enought annotator labels')
            1/0
        if len(pd['annotator_ids']) != 3:
            print('not enough annotator ids')
            1/0
        lbl, amount = collections.Counter(pd['annotator_labels']).most_common()[0]
        if lbl != pd['gold_label']:
            print('different gold labels')
            1/0
    print('Valid!')
    print('Save')
    with open(out_path, 'w') as f_out:
        for pd in parsed_data:
            f_out.write(json.dumps(pd) + '\n')

    print('Done')


def cohens_kappa(results, workers):
    """
    Compute Cohen's Kappa on all workers that answered at least 5 HITs
    :param results:
    :return:
    """
    answers_per_worker = { worker_id : { key : results[key][worker_id] for key in results.keys()
                                         if worker_id in results[key] }
                           for worker_id in workers }
    answers_per_worker = { worker_id : answers for worker_id, answers in answers_per_worker.items()
                           if len(answers) >= 5 }
    curr_workers = answers_per_worker.keys()
    worker_pairs = [(worker1, worker2) for worker1 in curr_workers for worker2 in curr_workers if worker1 != worker2]

    label_index = { 'contradiction' : 1, 'entailment' : 0 , 'neutral': 2}
    pairwise_kappa = { worker_id : { } for worker_id in answers_per_worker.keys() }

    # Compute pairwise Kappa
    for (worker1, worker2) in worker_pairs:

        mutual_hits = set(answers_per_worker[worker1].keys()).intersection(set(answers_per_worker[worker2].keys()))
        mutual_hits = set([hit for hit in mutual_hits]) # REMOVED:  if not pandas.isnull(hit)

        if len(mutual_hits) >= 5:

            worker1_labels = np.array([label_index[answers_per_worker[worker1][key][0]] for key in mutual_hits])
            worker2_labels = np.array([label_index[answers_per_worker[worker2][key][0]] for key in mutual_hits])
            curr_kappa = cohen_kappa_score(worker1_labels, worker2_labels)

            if not math.isnan(curr_kappa):
                pairwise_kappa[worker1][worker2] = curr_kappa
                pairwise_kappa[worker2][worker1] = curr_kappa

    # Remove worker answers with low agreement to others
    workers_to_remove = set()

    for worker, kappas in pairwise_kappa.items():
        if np.mean(list(kappas.values())) < 0.15:
            print('Removing', worker)
            workers_to_remove.add(worker)

    kappa = np.mean([k for worker1 in pairwise_kappa.keys() for worker2, k in pairwise_kappa[worker1].items()
                     if not worker1 in workers_to_remove and not worker2 in workers_to_remove])

    # Return the average
    return kappa, workers_to_remove

def eval_annotators(data_path):

    MIN_AMOUNT_SAME_HITS = 5
    MIN_AMOUNT_CO_WORKERS = 3

    not_enough_coworkers = []

    with open(data_path) as f_in:
        parsed_data = [json.loads(line.strip()) for line in f_in.readlines()]

    # create comparisons per annotator
    # => dict per annotator: ... only really use annotator id => HIT ids
    annotator_dict = collections.defaultdict(lambda: [])
    for pd in parsed_data:
        for i, annotator_id in enumerate(pd['annotator_ids']):
            annotator_dict[annotator_id].append((pd['annotator_labels'][i], pd))


    print('num annotators:', len(annotator_dict))

    hit_dict = collections.defaultdict(lambda: [])
    for pd in parsed_data:
        hit_dict[pd['hit_id']].append(pd)

    # find relevant matches:
    annotator_matches = []
    annotator_keys = sorted(list(annotator_dict.keys()))
    use_hits = []
    for i, annotator in enumerate(annotator_keys):
        annotator_hits = set([sample['hit_id'] for labeled, sample in annotator_dict[annotator]])

        # remember coworkers (annotator_id, hits)
        coworkers = []

        for j, other_annotator in enumerate(annotator_keys):

            if i != j:
                other_annotator_hits = set([sample['hit_id'] for labeled, sample in annotator_dict[other_annotator]])

                # find same HITs and check if enough
                same_hits = annotator_hits & other_annotator_hits
                if len(same_hits) >= MIN_AMOUNT_SAME_HITS:
                    coworkers.append((other_annotator, same_hits))

        # Only consider if enough workers
        if len(coworkers) >= MIN_AMOUNT_CO_WORKERS:
            annotator_matches.append((annotator, annotator_hits, coworkers))
            for _, same_hits in coworkers:
                use_hits.extend(list(same_hits))
        else:
            not_enough_coworkers.append(annotator)

    for annotator, annotator_hits, coworkers in annotator_matches:
        num_coworkers = len(coworkers)
        num_compare_amount_hits = sum([len(hits) for c, hits in coworkers])
        amount_annotator_hits = len(annotator_hits)

        print(annotator,':', 'num hits:', amount_annotator_hits, '; num coworkers:', num_coworkers, 'num compare hits:', num_compare_amount_hits)

    print('not comparing:', len(not_enough_coworkers), 'annotators due to not enough relevant coworkers')

    workers = set([annotator for annotator, annotator_hits, coworkers in annotator_matches])
    relevant_hits = set(use_hits)
    results = {}
    for pd in parsed_data:
        sample_id = pd['id']
        sample_annotators = pd['annotator_ids']
        sample_annotations = pd['annotator_labels']
        hit_id = pd['hit_id']

        if hit_id in relevant_hits:
            if sample_id not in results:
                results[sample_id] = {}

            for i in range(len(sample_annotations)):
                results[sample_id][sample_annotators[i]] = (sample_annotations[i], '')


    kappa, workers_to_remove = cohens_kappa(results, workers)
    print('kappa', workers_to_remove)
    print(kappa)

def postprocess(file_in, file_out):
    with open(file_in) as f_in:
        parsed_data = [json.loads(line.strip()) for line in f_in.readlines()]

    removed = 0
    used = 0
    renamed_category = 0
    translate_dict = dict([
        ('antonyms_wn', 'antonyms_wordnet'),
        ('numbers', 'cardinals'),
        ('nationalities_grouped', 'nationalities'),
        ('antonyms_other', 'antonyms'),
        ('countries_grouped', 'countries'),
        ('antonyms_adj_adv', 'antonyms'),
        ('antonyms_nn_vb', 'antonyms')
    ])

    def is_remove_sample(w1, w2):
        rm_set1 = set(['small', 'old'])
        rm_set2 = set(['little', 'old'])
        rm_set3 = set(['important', 'little'])

        if w1 in rm_set1 and w2 in rm_set1:
            return True
        if w1 in rm_set2 and w2 in rm_set2:
            return True
        if w1 in rm_set3 and w2 in rm_set3:
            return True
        if w1 == 'sand' and w2 != 'glass':
            return True
        return False

    with open(file_out, 'w') as f_out:
        for pd in parsed_data:
            new_json = {
                'category': pd['category'],
                'gold_label': pd['gold_label'],
                'annotator_labels': pd['annotator_labels'],
                'sentence1': pd['sentence1'],
                'sentence2': pd['sentence2'],
                'pairID': pd['id']
            }
            category = new_json['category']
            if category in translate_dict:
                new_json['category'] = translate_dict[category]
                renamed_category += 1
            if is_remove_sample(pd['replaced1'], pd['replaced2']):
                removed += 1
            else:
                used += 1
                f_out.write(json.dumps(new_json) + '\n')

    print('Done.')
    print('size:', used, ', removed:', removed, ', renamed:', renamed_category)

        
def subset(from_data, to_data, amount):
    prem_dict = collections.defaultdict(lambda: [])

    with open(from_data) as f_in:
        parsed_data = [json.loads(line.strip()) for line in f_in.readlines()]

    for pd in parsed_data:
        prem_dict[pd['sentence1']].append(pd)

    keep_samples = collections.defaultdict(lambda: [])

    cnt = 0
    for prem in prem_dict:
        if len(prem_dict[prem]) >= 5 and len(prem_dict[prem]) < 10:
            keep_samples[prem] = random.sample(prem_dict[prem], 5)
            cnt += 5
        elif len(prem_dict[prem]) >= 10 and len(prem_dict[prem]) < 15:
            keep_samples[prem] = random.sample(prem_dict[prem], 10)
            cnt += 10
        elif len(prem_dict[prem]) >= 15 and len(prem_dict[prem]) < 20:
            keep_samples[prem] = random.sample(prem_dict[prem], 15)
            cnt += 15
        elif len(prem_dict[prem]) >= 20 and len(prem_dict[prem]) < 25:
            keep_samples[prem] = random.sample(prem_dict[prem], 20)
            cnt += 20
        elif len(prem_dict[prem]) >= 25 and len(prem_dict[prem]) < 30:
            keep_samples[prem] = random.sample(prem_dict[prem], 25)
            cnt += 25
        elif len(prem_dict[prem]) >= 30 and len(prem_dict[prem]) < 35:
            keep_samples[prem] = random.sample(prem_dict[prem], 30)
            cnt += 30
        elif len(prem_dict[prem]) >= 35:
            print(len(prem_dict[prem]))

    print('total:', cnt)
    
    cnt = 0

    with open(to_data, 'w') as f_out:
        for p in keep_samples.keys():
            for pair in keep_samples[p]:
                f_out.write(json.dumps(pair) + '\n')
                cnt +=1

    print('Done:', cnt)

def recategorize(file_in, file_out):
    with open(file_in) as f_in:
        parsed = [json.loads(line.strip()) for line in f_in.readlines()]

    adapted = collections.defaultdict(int)
    neutral  = 0

    new_list = []
    samples = []
    for pd in parsed:
        if pd['category'] == 'synonyms' and pd['gold_label'] == 'neutral':
            neutral +=1

        if pd['gold_label'] == 'entailment':
            adapted[pd['category']] += 1
            if pd['category'] != 'synonyms':
                samples.append(pd)
            pd['category'] = 'entailing'


        new_list.append(pd)

    print(adapted.items())
    print(samples)
    print('neutral synonyms:', neutral)

    with open(file_out, 'w') as f_out:
        for pd in new_list:
            f_out.write(json.dumps(pd) + '\n')






def main():
    args = docopt("""Deal with data for mechanical turk.

    Usage:
        mechanical_turk_adversarial.py csv <file_in> <file_out>
        mechanical_turk_adversarial.py id <file_in> <file_out>
        mechanical_turk_adversarial.py filter_common <file_in> <file_out>
        mechanical_turk_adversarial.py verify <file_in>
        mechanical_turk_adversarial.py create_three <csv_in> <csv_out>
        mechanical_turk_adversarial.py result_json <csv_in> <file_out>
        mechanical_turk_adversarial.py finalize <results> <src> <out>
        mechanical_turk_adversarial.py finalize_wa <results> <src> <out>
        mechanical_turk_adversarial.py stats <new_dataset> <old_dataset>
        mechanical_turk_adversarial.py rm_incorrect <file_in> <file_out_correct> <file_out_incorrect>
        mechanical_turk_adversarial.py analyse <csv> <data> <result>
        mechanical_turk_adversarial.py kappa <dataset> <num_labels>
        mechanical_turk_adversarial.py create_sets <fulljson> <out>
        mechanical_turk_adversarial.py include_annotators <fulljson> <csv> <fout>
        mechanical_turk_adversarial.py eval_annotators <datapath>
        mechanical_turk_adversarial.py postprocess <file_in> <file_out>
        mechanical_turk_adversarial.py subset <file_in> <file_out> <amount>
        mechanical_turk_adversarial.py recategorize <file_in> <file_out>
    """)

    if args['csv']:
        csv(args['<file_in>'], args['<file_out>'])
    elif args['recategorize']:
        recategorize(args['<file_in>'], args['<file_out>'])
    if args['subset']:
        subset(args['<file_in>'], args['<file_out>'], int(args['<amount>']))
    elif args['postprocess']:
        postprocess(args['<file_in>'], args['<file_out>'])
    elif args['eval_annotators']:
        eval_annotators(args['<datapath>'])
    elif args['include_annotators']:
        include_annotator_ids(args['<fulljson>'], args['<csv>'], args['<fout>'])
    elif args['create_sets']:
        create_sets(args['<fulljson>'], args['<out>'])
    elif args['kappa']:
        kappa(args['<dataset>'], int(args['<num_labels>']))
    elif args['finalize_wa']:
        finalize_with_annotations(args['<results>'],args['<src>'], args['<out>'])
    elif args['analyse']:
        analyse(args['<csv>'], args['<data>'],args['<result>'])
    elif args['rm_incorrect']:
        remove_incorrect(args['<file_in>'], args['<file_out_correct>'], args['<file_out_incorrect>'])
    elif args['id']:
        id(args['<file_in>'], args['<file_out>'])
    elif args['filter_common']:
        filter_common(args['<file_in>'], args['<file_out>'])
    elif args['verify']:
        verify(args['<file_in>'])
    elif args['create_three']:
        # create three annotations by duplicating (per HIT)
        duplicate(args['<csv_in>'], args['<csv_out>'])
    elif args['result_json']:
        create_result_json(args['<csv_in>'], args['<file_out>'])
    elif args['finalize']:
        finalize(args['<results>'],args['<src>'], args['<out>'])
    elif args['stats']:
        stats(args['<new_dataset>'], args['<old_dataset>'])

if __name__ == '__main__':
    main()
