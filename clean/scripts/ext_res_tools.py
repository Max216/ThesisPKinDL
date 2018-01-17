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

def clean(res_to_clean, types):
    resources = [data_tools.ExtResPairhandler(path) for path in res_to_clean]
    known_types = ['syn', 'anto', 'hyp', 'cohyp']
    for t in types:
        if t not in known_types:
            print('Types must be one of the following:', ', '.join(known_types))
            return

    all_knowledge = dict()
    for i,res in enumerate(resources):
        for p, h, lbl in res.items():
            if p not in all_knowledge:
                all_knowledge[p] = dict()
            c_knowledge = all_knowledge[p]
            if h not in c_knowledge:
                c_knowledge[h] = [(lbl, types[i])]
            else:
                c_knowledge[h].append((lbl, types[i]))

    # now get conflicts
    conflicts = []
    for p in all_knowledge:
        c_knowledge = all_knowledge[p]
        for h in c_knowledge:
            c_set = set([origin for w, origin in c_knowledge[h]])
            h_set = set([w for w, origin in c_knowledge[h]])
            if len(h_set) != 1:
                conflicts.append((p, h, c_knowledge[h]))

    #print('Found the following conflicts:')
    #print('\n'.join([str(c) for c in conflicts]))

    # deal with conflicts
    type_to_res = dict([(types[i], resources[i]) for i in range(len(resources))])
    def deal_with_cohyp(p, h, conflicts):
        lbl_cohyp = [lbl for lbl, typ in conflicts if typ == 'cohyp'][0]
        if lbl_cohyp == 'contradiction':
            for lbl, typ in conflicts:
                if typ != 'cohyp' and lbl != 'contradiction':
                    type_to_res[typ].remove(p, h, lbl)
            return True

        # if entailment in hypernyms and neutral in cohyponyms, take entailment
        if len([(lbl, typ) for lbl, typ in conflicts if lbl == 'entailment' and typ == 'hyp']) >= 1:
            print('use as entailment')
            for lbl, typ in conflicts:
                if lbl != 'entailment':
                    type_to_res[typ].remove(p, h, lbl)
            return True

        # force remaining to neutral
        else:
            for lbl, typ in conflicts:
                if lbl != 'neutral':
                    type_to_res[typ].remove(p, h, lbl)
            return True

    print_out = 'anto'
    for p, h, results in conflicts:
        labels = [l for l, t in results]
        ctypes = [t for l, t in results]

        if print_out in ctypes:
            print(p, h, results)
        #print(p, h, results)

        dealt_with = False
        if 'cohyp' in ctypes:    
            dealt_with = deal_with_cohyp(p, h, results)

    print('Conflictss remaining', len(conflicts))
    print('Save updated resources')
    for i in range(len(resources)):
        resources[i].save(res_to_clean[i])

        




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
        ext_res_tools.py clean (-r <res_path>)...  (-t <res_types>)...

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
        data_tools.ExtResPairhandler(path_res, data_format=type_from).save(out_name, data_format=type_to)
    elif args['symmetric']:
        fn = symmetry_functions[symmetry_fn]
        ext_handler = data_tools.ExtResPairhandler(path_res)
        ext_handler.extend_from_own(extend_fn=fn)
        ext_handler.save(out_name)
    elif args['clean']:
        res_paths = args['<res_path>']
        res_types = args['<res_types>']
        if len(res_paths) != len(res_types):
            print('Must specify -t for each -r')
            return
        clean(res_paths, res_types)

if __name__ == '__main__':
    main()