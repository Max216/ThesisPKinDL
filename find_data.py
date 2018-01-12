'''
This script finds data and stored the found data
'''
import word_resource
import mydataloader
from docopt import docopt

def in_resource(res, p, h):
    return res.word_resource_overlap(p, h)
def no_filter(res, p, h):
    return True

def main():
    args = docopt("""Find data based on some criterion.

    Usage:
        find_data.py inres <data_path> <resource_path> <resource_label> <name_out>
        find_data.py valid <data_path> <name_out>

    """)

    data_path = args['<data_path>']
    resource_path = args['<resource_path>']
    resource_label = args['<resource_label>']
    name_out = args['<name_out>']

    filter_fn = None
    w_res = None
    if args['inres']:
        filter_fn = in_resource
        print('Create resource for:', resource_label, '...')
        w_res = word_resource.WordResource(resource_path, interested_relations=[resource_label])
        print('Loaded', len(w_res), 'word information.')
    elif args['valid']:
        filter_fn = no_filter

    print('Load relevant data ...')
    cnt_relevant = 0
    cnt_irrelevant = 0
    with open(data_path) as f_in:
        with open(name_out, 'w') as f_out:
            for line in f_in:
                p, h, lbl = mydataloader.extract_snli(line)
                if lbl in mydataloader.index_to_tag and filter_fn(w_res, p, h):
                    cnt_relevant += 1
                    f_out.write(line)
                else:
                    cnt_irrelevant += 1

    print('Loaded', cnt_relevant, 'samples. Not used:', cnt_irrelevant, 'samples.' )


if __name__ == '__main__':
    main()