from docopt import docopt
import gzip, os, re, collections

def main():
    args = docopt("""Create bigrams from google.

    Usage:
        create_bigrams <folder> <vocab_path> <out_path>
    """)

    folder = args['<folder>']
    vocab_path = args['<vocab_path>']
    out_path = args['<out_path>']

    # load vocab
    with open(vocab_path) as vocab_in:
        vocab = set([line.strip() for line in vocab_in.readlines()])

    bigram_dict = dict([(w, collections.defaultdict(int)) for w in vocab])

    split_regexp = r'\t'
    # go through all files
    count = 0
    for file in [os.path.join(folder, file) for file in os.listdir(folder) if file.startswith('googlebooks-eng-all-2gram-20120701')]:
        print('Open:', file)
        with gzip.open(file, 'r') as f_in:
            for line in f_in:
                line = line.decode('utf-8')
                splitted = re.split(split_regexp, line.strip())
                words = splitted[0].split(' ')
                
                word1 = words[0]
                word2 = words[1]

                splitted_w1 = word1.split('_')
                splitted_w2 = word2.split('_')

                if len(splitted_w1) > 1:
                    word1 = splitted_w1[0]
                if len(splitted_w2) > 1:
                    word2 = splitted_w2[0]

                #print(word1, word2)
                if len(word1) > 0 and len(word2) > 0 and word1 in vocab and word2 in vocab:
                    bigram_dict[word1][word2] +=  int(splitted[2])
                    count += 1
        print('Bigrams so far:', count)

    
    print('Write:', out_path)
    with open(out_path, 'w') as f_out:
        for key in bigram_dict:
            current_dict = bigram_dict[key]
            for key2 in current_dict:
                f_out.write('\t'.join([key,key2,str(current_dict[key2])]) + '\n')
        


if __name__ == '__main__':
    main()