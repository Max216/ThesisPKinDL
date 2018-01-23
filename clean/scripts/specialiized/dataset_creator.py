import sys, os
sys.path.append('./../')

from docopt import docopt

from libs import data_manipulator

def read_strp_lines(file):
    with open (file) as f_in:
        return [line.strip() for line in f_in]

def uknown_words(words):
    return [w for w in words if w in vocab]

def synonym_from_two_lists(list_a, list_b):
    return [(list_a[i], list_b[i], 'entailment') for i in range(len(list_a))]

def all_incompatible(words, exclude_words=[]):

    def is_excluded(w1, w2, exclude_words):
        for excl in exclude_words:
            if w1 in excl and w2 in excl:
                return True

        return False

    results = []
    for w1 in words:
        for w2 in words:
            if w1 != w2 and not is_excluded(w1, w2, exclude_words):
                results.append((w1, w2, 'contradiction'))

    return results

def incompatible_to_first(words1, words2):
    results = []
    for w1 in words1:
        for w2 in words2:
            results.append((w1, w2, 'contradiction'))
            results.append((w2, w1, 'contradiction'))

    return results



def countries():
    countries = 'America,China,India,England,Japan,Russia,Canada,Germany,Australia,Holland,France,Israel,Spain,Brazil,Jordan,Sweden,Greece,Italy,Ireland,Mexico,Switzerland,Singapore,Turkey,Ukraine,Egypt,Malaysia,Norway,Indonesia,Vietnam'.split(',')
    exclude_words = [set(['America', 'Canada'])]

    return ('countries', [], [], all_incompatible(countries, exclude_words=exclude_words))

def nationalities():
    nationalities = 'American,Chinese,English,Japanese,Russian,Canadian,Australian,Dutch,French,Israeli,Spanish,Brazilian,Jordanian,Swedish,Greek,Italian,Irish,Mexican,Swiss,Singaporean,Turkish,Ukrainian,Egyptian,Norwegian,Indonesian,Vietnamese'.split(',')
    exclude_words = [set(['American', 'Canadian'])]
    return ('nationalities', [], [], all_incompatible(nationalities, exclude_words=exclude_words))

def colors():
    colors = 'red,blue,yellow,purple,green,orange,brown,grey,black,white'.split(',')
    return ('colors', [], [], all_incompatible(colors))

def numbers():
    numbers_written = 'two,three,four,five,six,seven,eight,nine,ten,eleven,twelve'.split(',')
    numbers_digits = '2,3,4,5,6,7,8,9,10,11,12'.split(',')
    synonym_numbers = synonym_from_two_lists(numbers_written, numbers_digits)
    synonym_numbers.extend(synonym_from_two_lists(numbers_digits, numbers_written))
    return ('numbers', [], [], synonym_numbers)

def test():
    return ('test', [('a', 'HORSE', 'contradiction'), ('NOOO WAY', 'a', 'contradiction')], [('NOOO WAY', 'the', 'contradiction'), ('omelette', 'airplane', 'contradiction')], [('horse', 'omelette', 'contradiction')])

def test_out():
    words = 'red,blue,yellow,purple,green,orange,brown,grey,black,white'.split(',')
    datahandler = data_manipulator.DataManipulator().load()
    datahandler.print_sents(words, 20)
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
            colors
            #test
        ]

        datahandler = data_manipulator.DataManipulator().load()

        groups = []
        for fn in all_fn:
            name, replace_w1_only, replace_w2_only, replace_any = fn()

            generated_sample_holder = datahandler.generate_by_replacement(replace_w1_only, replace='w1')
            generated_sample_holder.merge(datahandler.generate_by_replacement(replace_w2_only, replace='w2'))
            generated_sample_holder.merge(datahandler.generate_by_replacement(replace_any, replace='any'))
            
            groups.append(name)
            directory = os.path.join(out_name, name)
            generated_sample_holder.write_summary(directory)
            generated_sample_holder.write_dataset(directory)

        with open(os.path.join(out_name, 'data.txt'), 'w') as f_out:
            for g in groups:
                f_out.write(g + '\n')

if __name__ == '__main__':
    main()
        

