from docopt import docopt

import json, re, collections, codecs, os
import csv as csv_lib


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
        mechanical_turk_adversarial.py stats <new_dataset> <old_dataset>

    """)

    if args['csv']:
        csv(args['<file_in>'], args['<file_out>'])
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
