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
        p['id'] = i
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

def clean_results(file_in, file_out):
    with open(file_in) as f_in:
        csv_reader = csv_lib.reader(f_in)
        content = [row for row in csv_reader]

    header = content[0]
    contents = content[1:]

    id_headers = ["Input.id1", "Input.id2","Input.id3","Input.id4","Input.id5"]
    question_1_headers = ["Answer.label1_hyp1","Answer.label1_hyp2", "Answer.label1_hyp3","Answer.label1_hyp4","Answer.label1_hyp5"]
    question_2_headers = ["Answer.label2_hyp1", "Answer.label2_hyp2", "Answer.label2_hyp3", "Answer.label2_hyp4", "Answer.label2_hyp5"]
    question_3_headers = ["Answer.nonsense_hyp1","Answer.nonsense_hyp2","Answer.nonsense_hyp3","Answer.nonsense_hyp4","Answer.nonsense_hyp5"]

    sample_id_indizes = [header.index(s_id) for s_id in id_headers]
    sample_q1_indizes = [header.index(q1) if q1 in header else -1 for q1 in question_1_headers]
    sample_q2_indizes = [header.index(q2) if q2 in header else -1 for q2 in question_2_headers]
    sample_q3_indizes = [header.index(q3) if q3 in header else -1 for q3 in question_3_headers]

   
    print(sample_id_indizes)
    print(sample_q1_indizes)
    print(sample_q2_indizes)
    print(sample_q3_indizes)

    with open(file_out, 'w') as f_out:
        pass
        1/0
        # TODO

def main():
    args = docopt("""Deal with data for mechanical turk.

    Usage:
        mechanical_turk_adversarial.py csv <file_in> <file_out>
        mechanical_turk_adversarial.py id <file_in> <file_out>
        mechanical_turk_adversarial.py filter_common <file_in> <file_out>
        mechanical_turk_adversarial.py clean_results <file_in> <file_out>

    """)

    if args['csv']:
        csv(args['<file_in>'], args['<file_out>'])
    elif args['id']:
        id(args['<file_in>'], args['<file_out>'])
    elif args['filter_common']:
        filter_common(args['<file_in>'], args['<file_out>'])
    elif args['clean_results']:
        clean_results(args['<file_in>'], args['<file_out>'])

if __name__ == '__main__':
    main()
