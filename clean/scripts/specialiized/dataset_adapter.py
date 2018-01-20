import sys
sys.path.append('./../')

from docopt import docopt

from libs import data_tools

def create_incompatible_pairs(incompatible_words):
    samples = []
    for w1 in incompatible_words:
        for w2 in incompatible_words:
            if w1 != w2:
                samples.append((w1, w2, 'contradiction'))

    return samples

def create_hypernym_pairs(hypernyms, hyponyms):
    samples = []
    for hyper in hypernyms:
        for hypo in hyponyms:
            samples.append((hypo, hyper, 'entailment'))
            samples.append((hyper, hypo, 'neutral'))

    return samples

def create_synonyms(w1, syns, symmetric=False):
    samples = []
    for syn in syns:
        samples.append((w1, syn, 'entailment'))
        if symmetric:
            samples.append((syn, w1, 'entailment'))
    return samples



def create_list1():
    list_countries1 = 'USA Israel Canada Italy Spain France England Germany'.split(' ')
    list_countries2 = 'American Israeli Canadian Italian Spanish French English German'.split(' ')
    list_animals_cont = 'elephant tiger lion bee frog snake cow bird horse spider butterfly cat dog'.split(' ')
    list_animals_hyper = ['animal']
    list_insects_hypo = 'bee butterfly'.split(' ')
    list_insects_hyper = ['insect']

    samples = create_incompatible_pairs(list_countries1)
    samples.extend(create_incompatible_pairs(list_countries2))
    samples.extend(create_incompatible_pairs(list_animals_cont))
    samples.extend(create_hypernym_pairs(list_animals_hyper, list_animals_cont))
    samples.extend(create_hypernym_pairs(list_insects_hyper, list_insects_hypo))
    return samples

def create_list2():
    hyper1 = create_hypernym_pairs(['partner'], 'boyfriend girlfriend wife husband'.split(' '))
    hyper2 = create_hypernym_pairs(['person'], 'chemist teacher artist singer expert student'.split(' '))
    hyper3 = create_hypernym_pairs(['relative'], 'brother sister mother father cousin'.split(' '))

    syn1 = create_synonyms('big', ['huge'], symmetric=True)
    syn2 = create_synonyms('big', 'great', symmetric=False)
    syn3 = create_synonyms('destroy', ['demolish'], symmetric=True)
    syn4 = create_synonyms('happy', ['cheerful', 'delighted'], symmetric=True)
    syn5 = create_synonyms('quickly', ['rapidly'], symmetric=True)
    syn6 = create_synonyms('quickly', ['fast'], symmetric=False)
    syn7 = create_synonyms('hurry', ['rush'], symmetric=True)
    syn8 = create_synonyms('important', ['essential', 'necessary'], symmetric=True)
    syn9 = create_synonyms('afraid', ['scared'], symmetric=True)

    incompatible1 = create_incompatible_pairs('girlfriend boyfriend'.split(' '))
    incompatible2 = create_incompatible_pairs('wife husband'.split(' '))
    incompatible3 = create_incompatible_pairs('brother sister'.split(' '))
    incompatible4 = create_incompatible_pairs('father mother'.split(' '))

    return hyper1 + hyper2 + hyper3 + syn1 + syn2 + syn3 + syn4 + syn5 + syn6 + syn7 + syn8 + syn9 + incompatible1 + incompatible2 + incompatible3 + incompatible4


def main():
    args = docopt("""Create a data file from arrays.

    Usage:
        dataset_adapter.py list1 <out_name>
        dataset_adapter.py list2 <out_name>
    """)

    out_name = args['<out_name>']
    print(out_name)
    samples = None

    if args['list1']:
        samples = create_list1()
    elif args['list2']:
        samples = create_list2() 

    res_handler = data_tools.ExtResPairhandler()
    res_handler.add(samples)
    res_handler.save(out_name)



if __name__ == '__main__':
    main()