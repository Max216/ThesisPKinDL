from docopt import docopt

import json, re, collections, codecs, os
import csv as csv_lib
import numpy as np


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
                        category = sample['category']
                        replaced1 = sample['replaced1']
                        replaced2 = sample['replaced2']
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
        p['id'] = i + 10000
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

def kappa_from_samples(samples, num_categories = 4):
    label_dict = dict([('entailment', 0), ('contradiction', 1), ('neutral', 2), ('incorrect', 3)])
    num_items = len(samples)
    matrix = np.zeros((num_items, num_categories))

    for i,s in enumerate(samples):
        for annotator_label in s['annotator_labels']:
            matrix[i, label_dict[annotator_label]] += 1

    return fleiss_kappa(matrix)

def kappa(results_path, data_path):
    num_categories = 4


    with open(data_path) as f_in:
        parsed_data = [json.loads(line.strip()) for line in f_in.readlines()]
    cat_dict = dict([(str(p['id']), p['category']) for p in parsed_data])

    with open(results_path) as f_in:
        parsed = [json.loads(line.strip()) for line in f_in.readlines()]
        annotations = [p['labels'] for p in parsed]
        annotation_dict = dict([(str(p['id']), p['labels']) for p in parsed])

    label_dict = dict([('entailment', 0), ('contradiction', 1), ('neutral', 2), ('incorrect', 3)])

    num_items = 10000

    assert len(annotations) == num_items

    matrix = np.zeros((num_items, num_categories))
    for i, labels in enumerate(annotations):
        for lbl in labels:
            matrix[i, label_dict[lbl]] += 1

    assert np.sum(matrix) == 10000 * 3

    print('Fleiss Kappa all:', fleiss_kappa(matrix))

    all_categories = list(set([cat for k, cat in cat_dict.items()]))


    total_cnt_1_label = 0
    total_cnt_2_label = 0
    total_cnt_3_label = 0
    for cat in all_categories:
        cnt_1_label = 0
        cnt_2_label = 0
        cnt_3_label = 0
        for k, lbls in annotation_dict.items():
            if cat_dict[k] == cat:
                num_annotations = len(set(lbls))
                if num_annotations == 1:
                    cnt_1_label += 1
                elif num_annotations == 2:
                    cnt_2_label += 1
                else:
                    cnt_3_label += 1

        print('In', cat, ':')
        print('All same label:', cnt_1_label)
        print('Two different labels:', cnt_2_label)
        print('Three different labels:', cnt_3_label)
        print()
        total_cnt_1_label += cnt_1_label
        total_cnt_2_label += cnt_2_label
        total_cnt_3_label += cnt_3_label

    print('In total:')
    print('All same label:', total_cnt_1_label)
    print('Two different labels:', total_cnt_2_label)
    print('Three different labels:', total_cnt_3_label)


def analyse_full_json(path_json):
    with open(path_json) as f_in:
        parsed = [json.loads(line.strip()) for line in f_in.readlines()]

    print('# General stats:')
    print('# Labels')
    print(collections.Counter([p['gold_label'] for p in parsed]).most_common())

    # Filtered
    parsed_valid = [p for p in parsed if p['gold_label'] != '-' and p['gold_label'] != 'incorrect']
    print('Valid samples:', len(parsed_valid))

    print('# Categories')
    print(collections.Counter([p['category'] for p in parsed_valid]).most_common())

    print('# Fleiss')
    print('Over all data with used categories: size =',len(parsed),': ', kappa_from_samples(parsed))
   


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
        mechanical_turk_adversarial.py kappa <results> <data>
        mechanical_turk_adversarial.py anl <fulljson>
        mechanical_turk_adversarial.py create_sets <fulljson> <out>

    """)

    if args['csv']:
        csv(args['<file_in>'], args['<file_out>'])
    elif args['create_sets']:
        create_sets(args['<fulljson>'], args['<out>'])
    elif args['anl']:
        analyse_full_json(args['<fulljson>'])
    elif args['kappa']:
        kappa(args['<results>'], args['<data>'])
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
