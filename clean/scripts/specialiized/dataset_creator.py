import sys, os
sys.path.append('./../')

from docopt import docopt

from libs import data_manipulator

def read_strp_lines(file):
    with open (file) as f_in:
        return [line.strip() for line in f_in]

def is_excluded(w1, w2, exclude_words):
        for excl in exclude_words:
            if w1 in excl and w2 in excl:
                return True

        return False

def uknown_words(words):
    return [w for w in words if w in vocab]

def synonym_from_two_lists(list_a, list_b):
    return [(list_a[i], list_b[i], 'entailment') for i in range(len(list_a))]

def all_incompatible(words, exclude_words=[]):

    results = []
    for w1 in words:
        for w2 in words:
            if w1 != w2 and not is_excluded(w1, w2, exclude_words):
                results.append((w1, w2, 'contradiction'))

    return results

def incompatible_to_first(words1, words2, exclude_words=[], symmetric=True):
    results = []
    for w1 in words1:
        for w2 in words2:
            if not is_excluded(w1, w2, exclude_words):
                results.append((w1, w2, 'contradiction'))
                if symmetric:
                    results.append((w2, w1, 'contradiction'))

    return results

def synonyms():
    two_way_synonyms1 = 'starts,quickly,noisy,no one,angry,angry,aweful,dangerous,beautiful,fantastic,enormous,intelligent,wealthy,famous,stupid,interesting,a lot of'.split(',')
    two_way_synonyms2 = 'begins,rapidly,loud,nobody,furious,mad,terrible,risky,pretty,wonderful,huge,clever,rich,well-known,dumb,fascinating,plenty of'.split(',')

    syn_group1 = 'happy,pleased,delighted,joyful,glad'.split(',')
    syn_group2 = 'sad,miserable,unhappy'.split(',')
    syn_group3 = 'excellent,outstanding,phenomenal'.split(',')

    replace_only_first = 'begin,quickly,rapidly,silent,crazy,ancient,a different'.split(',')
    not_replace_second = 'start,fast,fast,quiet,mad,old,the same'.split(',')

def countries():
    countries = 'America,China,India,England,Japan,Russia,Canada,Germany,Australia,Holland,France,Israel,Spain,Brazil,Sweden,Greece,Italy,Ireland,Mexico,Switzerland,Singapore,Turkey,Ukraine,Egypt,Malaysia,Norway,Vietnam'.split(',')
    exclude_words = [set(['America', 'Canada'])]

    return ('countries', [], [], all_incompatible(countries, exclude_words=exclude_words))

def nationalities():
    exclude_words = []

    nationalities_same_singular_plural = 'Chinese,English,Japanese,Dutch,French,Spanish,Swedish,Irish,Turkish,Vietnamese'.split(',')
    nationalities_different_plural_singular = 'Russian,Canadian,German,Australian,Israeli,Brazilian,Greek,Italian,Mexican,Ukrainian,Egyptian,Norwegian,Indonesian'.split(',')
    nationalities_different_plural_plural = 'Russians,Canadians,Germans,Australians,Israelis,Brazilians,Greeks,Italians,Mexicans,Ukrainians,Egyptians,Norwegians,Indonesians'.split(',')
    
    replace_any = all_incompatible(nationalities_same_singular_plural)
    replace_any.extend(all_incompatible(nationalities_different_plural_plural))
    replace_any.extend(all_incompatible(nationalities_different_plural_singular))

    return ('nationalities', [], [], all_incompatible(nationalities, exclude_words=exclude_words))

def colors():
    colors = 'red,blue,yellow,purple,green,orange,brown,grey,black,white'.split(',')
    return ('colors', [], [], all_incompatible(colors))

def numbers():
    numbers_written = 'two,three,four,five,six,seven,eight,nine,ten,eleven,twelve'.split(',')
    numbers_digits = '2,3,4,5,6,7,8,9,10,11,12'.split(',')

    exclude_pairs = [set([numbers_written[i], numbers_digits[i]]) for i in range(len(numbers_written))]

    incompatible_written_numbers = all_incompatible(numbers_written)
    incompatible_digits = all_incompatible(numbers_digits)
    incompatible_written_premise = incompatible_to_first(numbers_written, numbers_digits, exclude_pairs, False)
    incompable_digits_premise = incompatible_to_first(numbers_digits, numbers_written, exclude_pairs, False)

    synonym_numbers = synonym_from_two_lists(numbers_written, numbers_digits)
    synonym_numbers_reversed = synonym_from_two_lists(numbers_digits, numbers_written)

    replace_first = synonym_numbers + incompatible_written_premise
    replace_second = synonym_numbers_reversed + incompable_digits_premise
    replace_any = incompatible_written_numbers + incompatible_digits


    return ('numbers', replace_first, replace_second, replace_any)

def test():
    return ('test', [('a', 'HORSE', 'contradiction'), ('NOOO WAY', 'a', 'contradiction')], [('NOOO WAY', 'the', 'contradiction'), ('omelette', 'airplane', 'contradiction')], [('horse', 'omelette', 'contradiction')])

def test_out():
    words = 'Russians,Canadians,Germans,Australians,Israelis,Brazilians,Greeks,Italians,Mexicans,Ukrainians,Egyptians,Norwegians,Indonesians'.split(',')
    datahandler = data_manipulator.DataManipulator().load()
    datahandler.print_sents(words, 30)
def main():
    args = docopt("""Create a new dataset based on the given type.

    Usage:
        dataset_creator.py create <out_name>
        dataset_creator.py test 
        dataset_creator.py show -a <amount> (-w <words>)...
    """)


    if args['test']:
        test_out()
    elif args['show']:
        max_amount = int(args['<amount>'])
        words = args['<words>']
        datahandler = data_manipulator.DataManipulator().load()
        datahandler.print_sents(words, max_amount)
    else:
        out_name = args['<out_name>']
        all_fn = [
            countries,
            nationalities,
            colors,
            numbers
            #test
        ]

        datahandler = data_manipulator.DataManipulator().load()

        groups = []
        for fn in all_fn:
            name, replace_w1_only, replace_w2_only, replace_any = fn()

            generated_sample_holder = datahandler.generate_simply(replace_w1_only, replace='w1')
            generated_sample_holder.merge(datahandler.generate_simply(replace_w2_only, replace='w2'))
            generated_sample_holder.merge(datahandler.generate_simply(replace_any, replace='any'))
            
            groups.append(name)
            directory = os.path.join(out_name, name)
            generated_sample_holder.write_summary(directory)
            generated_sample_holder.write_dataset(directory)

        with open(os.path.join(out_name, 'data.txt'), 'w') as f_out:
            for g in groups:
                f_out.write(g + '\n')

if __name__ == '__main__':
    main()
        

