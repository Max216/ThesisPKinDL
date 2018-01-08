import mydataloader

def build_from_snli_format(path, relations):
    '''
    Load the resource from json-SNLI format.
    '''

    resource_dict = dict()

    with open(path) as f_in:
        for line in f_in:
            line = line.strip()
            w1, w2, relation = mydataloader.extract_snli(line)
            if relation in relations:
                if w1 not in resource_dict:
                    resource_dict[w1] = dict()

                if w2 in resource_dict[w1]:
                    print('[Warning]', 'Overwriting', w1, '-', w2, '(' + resource_dict[w1][w2] + ') with', relation)
                resource_dict[w1][w2] = relation

    return resource_dict

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
            self.resource_dict = build_from_snli_format(res_path, interested_relations)

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
            cnt += len(resource_dict[w1])

        return cnt



