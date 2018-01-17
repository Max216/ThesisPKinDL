import sys
sys.path.append('./../') 

from docopt import docopt
 
from libs import data_tools

def filter(vocab, res_handler, out_name):
    res_handler.filter_vocab(vocab)
    res_handler.save(out_name)

def read_strp_lines(file):
    with open (file) as f_in:
        return [line.strip() for line in f_in]

def clean(res_to_clean):
    resources = [data_tools.ExtResPairhandler(path) for path in res_to_clean]
    all_knowledge = dict()
    for res in resources:
        for p, h, lbl in res.items():
            if p not in all_knowledge:
                all_knowledge[p] = dict()
            c_knowledge = all_knowledge[p]
            if h not in c_knowledge:
                c_knowledge[h] = [lbl]
            else:
                c_knowledge[h].append(lbl)

    # now get conflicts
    conflicts = []
    for p in all_knowledge:
        c_knowledge = all_knowledge[p]
        for h in c_knowledge:
            if len(c_knowledge[h]) != 1:
                conflicts.append((p, h, c_knowledge[h]))

    print('Found the following conflicts:')
    print(conflicts)




def symmetry_fn_keep_label(p, h, label):
    return (h, p, label)

def symmetry_fn_entailment_to_neutral(p, h, label):
    if label == 'entailment':
        return (h, p, 'neutral')
    else:
        return (h, p, label)

symmetry_functions = dict([('same_label', symmetry_fn_keep_label), ('e_to_n', symmetry_fn_entailment_to_neutral)])

def main():
    args = docopt("""Preprocess external resource files.

    Usage:
        ext_res_tools.py filter <vocab> <ext_res> <out_name>
        ext_res_tools.py convert <ext_res> <type_from> <type_to> <out_name>
        ext_res_tools.py symmetric <ext_res> <symmetry_fn> <out_name>
        ext_res_tools.py clean (-r <res_path>)...

    """)

    path_vocab  = args['<vocab>']
    path_res  = args['<ext_res>']
    out_name  = args['<out_name>']
    symmetry_fn = args['<symmetry_fn>']

    

    if args['filter']:
        ext_handler = data_tools.ExtResPairhandler(path_res)
        filter(read_strp_lines(path_vocab), ext_handler, out_name)
    elif args['convert']:
        type_from = args['<type_from>']
        type_to = args['<type_to>']
        data_tools.ExtResPairhandler(path_res, data_format='txt_01_nc').save(out_name, data_format='snli')
    elif args['symmetric']:
        fn = symmetry_functions[symmetry_fn]
        ext_handler = data_tools.ExtResPairhandler(path_res)
        ext_handler.extend_from_own(extend_fn=fn)
        ext_handler.save(out_name)
    elif args['clean']:
        clean(args['<res_path>'])

if __name__ == '__main__':
    main()