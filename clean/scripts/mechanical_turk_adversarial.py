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
        f_out.write('premise,hypothesis1,hypothesis2,hypothesis3,hypothesis4,hypothesis5\n')
        for premise in premise_dict:
            current = premise_dict[premise]

            chunks = [current[x:x+5] for x in range(0, len(current), 5)]


            for chunk in chunks:
                current_hit = []
                premise = None
                for i, sample in enumerate(chunk):
                    premise = sample['sentence1']
                    hypothesis = sample['sentence2']
                    label = sample['gold_label']
                    category = sample['category']
                    replaced1 = sample['replaced1']
                    replaced2 = sample['replaced2']

                    rep2_regexp = re.compile('\\b' + replaced2 + '\\b')
                    split_hyp = re.split(rep2_regexp, hypothesis)
                    split_prem = re.split(rep2_regexp, premise)

                    new_hyp = ("<span class='highlight'>" + replaced2 + '</span>').join(split_hyp)
                    current_hit.append(new_hyp.replace(',', '&#44;').replace('\"', '&quot;'))


                out = [premise.replace(',', '&#44;').replace('\"', '&quot;')]
                for sample in chunk:
                    new_out = out + current_hit
                    f_out.write(u','.join(new_out) + os.linesep)


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


def main():
    args = docopt("""Deal with data for mechanical turk.

    Usage:
        mechanical_turk_adversarial.py csv <file_in> <file_out>
        mechanical_turk_adversarial.py id <file_in> <file_out>

    """)

    if args['csv']:
        csv(args['<file_in>'], args['<file_out>'])
    elif args['id']:
        id(args['<file_in>'], args['<file_out>'])

if __name__ == '__main__':
    main()