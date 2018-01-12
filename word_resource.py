import mydataloader

def build_from_snli_format(path, relations, max_words=1):
    '''
    Load the resource from json-SNLI format.
    '''

    resource_dict = dict()

    cnt_relevant = 0
    cnt_irrelevant = 0

    with open(path) as f_in:
        for line in f_in:
            line = line.strip()
            w1, w2, relation = mydataloader.extract_snli(line)
            if len(w1) == max_words and len(w2) == max_words:
                cnt_relevant += 1
                w1 = w1[0]
                w2 = w2[0]
                if relation in relations:
                    if w1 not in resource_dict:
                        resource_dict[w1] = dict()

                    if w2 in resource_dict[w1]:
                        print('[Warning]', 'Overwriting', w1, '-', w2, '(' + resource_dict[w1][w2] + ') with', relation)
                    resource_dict[w1][w2] = relation
            else:
                cnt_irrelevant +=1
                #print('[Warning]', 'More than one word in resource:', w1, w2)

    print('Dictionary:', 'use', cnt_relevant, '; don\'t use:', cnt_irrelevant, 'samples with label:', relations)
    return resource_dict

def build_single_pair(data):
    w1 = data[0]
    w2 = data[1]
    relation = data[2]

    return dict([(w1, dict([(w2, relation)]))])

class WordResource:
    '''
    Store information about pairwise words from a resouce
    '''

    def __init__(self, res_path, build_fn='snli', interested_relations=['contradiction', 'entailment']):
        '''
        Create a new resource

        :param res_path 	Path to data of resource
        :param interested_relations		Only load those samples into the resource
        :param build_fn 	function(res_path, interested_relations) to create from a file. Default uses SNLI snyntax
        '''
        self.relations = interested_relations

        if build_fn == 'snli':
            self.resource_dict = build_from_snli_format(res_path, interested_relations, max_words=1)
        elif build_fn == 'single_pair':
            self.resource_dict = build_single_pair(res_path)

    def word_resource_overlap(self, sent1, sent2):
        '''
        Check if the two sentences (in this order) each contain at least one word, s.t. there is a relation between
        those two words stored within this resource.

        :param sent1 tokenized sentence 1 (premise)
        :param sent2 tokenzied sentence 2 (hypothesis)
        '''

        for w1 in sent1:
            for w2 in sent2:
                if w1 in self.resource_dict and w2 in self.resource_dict[w1]:
                    return True

        return False

    def __len__(self):
        cnt = 0
        for w1 in self.resource_dict:
            cnt += len(self.resource_dict[w1])

        return cnt

    def get_word_pairs(self, sent1, sent2):
        '''
        Get all pairs of (w1 w2) where w1 is in sent1 and w2 is in sent2 and both of them are in this resource.
        Instead of the words, indizes are returned.
        '''

        results = []
        for idx1, w1 in enumerate(sent1):
            if w1 in self.resource_dict:
                current = self.resource_dict[w1]
                for idx2, w2 in enumerate(sent2):
                    if w2 in current:
                        results.append((idx1, idx2))

        return results




