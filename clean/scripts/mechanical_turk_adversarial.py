from docopt import docopt

import json, re, collections, codecs, os


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



def main():
    args = docopt("""Deal with data for mechanical turk.

    Usage:
        mechanical_turk_adversarial.py csv <file_in> <file_out>
        mechanical_turk_adversarial.py id <file_in> <file_out>
        mechanical_turk_adversarial.py filter_common <file_in> <file_out>

    """)

    if args['csv']:
        csv(args['<file_in>'], args['<file_out>'])
    elif args['id']:
        id(args['<file_in>'], args['<file_out>'])
    elif args['filter_common']:
        filter_common(args['<file_in>'], args['<file_out>'])

if __name__ == '__main__':
    main()
