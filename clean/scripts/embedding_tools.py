'''
To deal with embeddings
'''

import sys
sys.path.append('./../')

from docopt import docopt
 
from libs import data_tools, data_handler
import nltk
from nltk.corpus import wordnet as wn

def diff(embeddings1, embeddings2):


    with open(embeddings1) as f1:
        vocab1 = set([line.split(' ')[0] for line in f1.readlines()])

    with open(embeddings1) as f2:
        vocab2 = set([line.split(' ')[0] for line in f2.readlines()])

    only1 = vocab1 - vocab2
    print('In first:', len(only1), 'distinct')
    print(only1)

    only2 = vocab2 - vocab1
    print('In 2nd:', len(only2), 'distinct')
    print(only2)



def cfd(embedding_path, data1_path, data2_path, data3_path, name_out):
    datahandler1 = data_handler.Datahandler(data1_path)
    datahandler2 = data_handler.Datahandler(data2_path)
    datahandler3 = data_handler.Datahandler(data3_path)
    datahandler1.merge([datahandler2, datahandler3])
    vocab = datahandler1.vocab()

    print('Total vocab in files:', len(vocab))
    with open(embedding_path) as f_in:
        used_embeddings = [line for line in f_in if line.split(' ', maxsplit=2)[0] in vocab]

    print('Word embeddings for vocab:', len(used_embeddings))
    with open(name_out, 'w') as f_out:
        for line in used_embeddings:
            f_out.write(line)

def concat_hypernyms(embedding_file, all_embeddings, path_out):

    print('Load embeddings within SNLI')
    with open(embedding_file) as f_in:
        stored_embeddings = [line.rstrip() for line in f_in.readlines()]

    print('Load all embeddings')
    all_embeddings_dict = dict()
    with open(all_embeddings, 'rb') as f_in:
        for line in f_in:
            entries = line.strip().split(b' ')
            word, entries = entries[0], entries[1:]
            try:
                if isinstance(word, six.binary_type):
                    word = word.decode('utf-8')
            except:
                print('non-UTF8 token', repr(word), 'ignored')
                continue

            all_embeddings_dict[word] = ' '.join([val.decode('utf-8') for val in entries])

    print('Prepocess WordNet')
    hyper = lambda s: s.hypernyms()

    print('Find hypernyms')
    results = []
    with open(path_out, 'w') as f_out:
        for stored in stored_embeddings:
            word = stored.split(' ')[0]
            synsets = wn.synsets(word, pos=wn.NOUN)
            if len(synsets) > 0:

                # Just use first synset
                syns = synsets[0]

                # Find hypernyms
                hypernyms = [hyp for hyp in syns.closure(hyper, depth=1)]
                if len(hypernyms) > 0:
                    lemmas = [lemma for lemma in hypernyms[0].lemmas() if len(lemma.split(' ')) == 1]

                    if len(lemmas) > 0:
                        # check if it is in embeddings
                        for lemma in lemmas:
                            if lemma in all_embeddings_dict:
                                results.append(word + ' ' + all_embeddings_dict[lemma] + '\n')
                                added = True
                                break

        print('Found', len(results), 'hypernyms')
        for line in results:
            f_out.print(line)




            

def main():
    args = docopt("""Deal with embeddings. 
        cfd  = create for data: Create embedding files for a given dataset.

    Usage:
        embedding_tools.py cfd <embeddings> <data_train> <data_dev> <data_test> <name_out>
        embedding_tools.py diff <embeddings1> <embeddings2>
        embedding_tools.py concat_hypernyms <embedding_file> <all_embeddings> <name_out>

    """)

    embeddings  = args['<embeddings>']
    data_train  = args['<data_train>']
    data_dev  = args['<data_dev>']
    data_test  = args['<data_test>']
    name_out  = args['<name_out>']

    if args['cfd']:
        cfd(embeddings, data_train, data_dev, data_test, name_out)
    elif args['diff']:
        diff(args['<embeddings1>'], args['<embeddings2>'])
    elif args['concat_hypernyms']:
        concat_hypernyms(args['<embedding_file>'], args['<all_embeddings>'], args['<name_out>'])


if __name__ == '__main__':
    main()