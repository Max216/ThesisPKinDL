import sys
sys.path.append('./../../')

from docopt import docopt

from bs4 import BeautifulSoup
import requests

from libs import data_tools

def read_strp_lines(file):
    with open (file) as f_in:
        return [line.strip() for line in f_in]

def symmetry_fn_keep_label(p, h, label):
    return (h, p, label)

def get_antonym1_samples():
    url = 'http://www.enchantedlearning.com/wordlist/opposites.shtml'
    print('Retrieve antonyms from', url)
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    tds = soup.find_all('td')
    tds = [td for td in tds if len(td.find_all('a')) == 0]
    samples = []
    for td in tds[2:-1]:
        lines = td.text.strip().split('\n')
        for line in lines:
            splitted = line.split(' - ')
            if len(splitted) != 2:
                # ignore
                continue
            w1 = splitted[0].strip()
            all_w2 = [w.strip() for w in splitted[1].split(',')]
            samples.extend([(w1, w2, 'contradiction') for w2 in all_w2])

    return samples


def main():
    args = docopt("""Create a data file from a web resource.

    Usage:
        wordlist_scraper.py antonyms1 <vocab> <out_name>
    """)

    vocab_path  = args['<vocab>']
    out_name  = args['<out_name>']

    symmetric = False
    if args['antonyms1']:
        symmetric = True
        samples = get_antonym1_samples()

    res_handler = data_tools.ExtResPairhandler().init_with_samples(samples)
    print('Retrieved', len(res_handler), 'samples.')

    # check with vocab
    vocab = read_strp_lines(vocab_path)
    res_handler.filter_vocab(vocab)
    print(len(res_handler), 'samples remaining. (only keep those in vocabulary)')

    # check if at least one sample contains both words
    keep_order = not symmetric
    datahandler = data_tools.get_datahandler_train()
    res_handler.filter_data(datahandler, keep_order=keep_order, req_label='contradiction')
    print(len(res_handler), 'samples remaining. (only keep those in train data)')

    # make symmetric
    if symmetric:
        res_handler.extend_from_own(symmetry_fn_keep_label)
        print(len(res_handler), 'samples after using symmetry.')

    # save
    res_handler.save(out_name)



if __name__ == '__main__':
    main()