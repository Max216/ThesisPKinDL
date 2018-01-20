'''To evaluate word pairs'''

from libs import evaluate, data_tools

import os

def stringify(arr):
    return ' '.join([str(v) for v in arr])

def intify(line):
    return [int(v) for v in line.split(' ') if v != '']

class AdvEvaluator:
    '''
    Evaluates adversarial sampels
    '''

    TYPE_SWAP_ANY = 'swap any'
    TYPE_SWAP_W1_TO_W2 = 'w1 > w2'
    TYPE_SWAP_W2_TO_W1 = 'w2 > w1'

    def __init__(self, path=None):
        '''
        Create a new object to gernate the given adversarial samples
        :path       Path where to load information from.
        '''
        self.evaluated = False
        if path != None:
            with open(path) as f_in:
                self._load([line.strip() for line in f_in.readlines()])
            
            self.evaluated = True

        self.generation_types = [self.TYPE_SWAP_ANY, self.TYPE_SWAP_W1_TO_W2, self.TYPE_SWAP_W2_TO_W1]

    def generation_types(self):
        return self.generation_types

    def evaluate(self, classifier, adv_sample_handler, datahandler, embedding_holder):
        '''
        Generates the evaluation of the classifier for the given samples.
        :param classifier           classifier to check
        :param adv_sample_handler   contains information about adversarial samples
        :param datahandler          contains the actual data (ALL data that was used to create adv_sample_handler)
        '''

        self.adv_sample_handler = adv_sample_handler
        self.datahandler = datahandler

        # evaluate natural samples

        # predictions per label per label
        # including samples per label per label
        labels = datahandler.valid_labels
        self.natural_sample_count = dict([(lbl, dict([(lbl2, 0) for lbl2 in labels])) for lbl in labels])
        self.natural_samples = dict([(lbl, dict([(lbl2, []) for lbl2 in labels])) for lbl in labels])
        for gold_label in adv_sample_handler.valid_labels:
            natural_dataset_for_label = adv_sample_handler.get_natural_dataset(datahandler, embedding_holder, gold_label)
            
            identifiers = [i for i in range(len(natural_dataset_for_label))]
            pred_dict, ident_dict = evaluate.create_prediction_dict(classifier, natural_dataset_for_label, embedding_holder.padding(), labels, identifiers=identifiers)
            
            for predicted_label in pred_dict[gold_label]:
                self.natural_sample_count[gold_label][predicted_label] = pred_dict[gold_label][predicted_label]
                self.natural_samples[gold_label][predicted_label] = ident_dict[gold_label][predicted_label]

        # evaluate adversarial samples
        adversarial_dataset = adv_sample_handler.get_adversarial_dataset(datahandler, embedding_holder)
        identifiers = [i for i in range(len(adversarial_dataset))]
        pred_dict, ident_dict = evaluate.create_prediction_dict(classifier, adversarial_dataset, embedding_holder.padding(), labels, identifiers=identifiers)
        
        # All must have same gold label.
        self.adversarial_sample_count = pred_dict[self.adv_sample_handler.label]
        self.adversarial_samples = ident_dict[self.adv_sample_handler.label]
        
        self.evaluated = True

    def _get_data_ids_for_predicted(self, generation_type):
        if generation_type == self.TYPE_SWAP_W1_TO_W2:
            return self.adv_sample_handler.get_dataset_sample_idx_w1_replaced()
        elif generation_type == self.TYPE_SWAP_W2_TO_W1:
            return self.adv_sample_handler.get_dataset_sample_idx_w2_replaced()
        else:
            return self.adv_sample_handler.get_dataset_sample_idx_any_replaced()


    def adversarial_prediction_dict(self, generation_type):
        if generation_type not in self.generation_types:
            print('must have on of the following generation types', self.generation_types)
            1/0

        if generation_type == self.TYPE_SWAP_ANY:
            return self.adversarial_sample_count
        else:
            ident_indizes = [idx for idx, _ in self._get_data_ids_for_predicted(generation_type)]
            #print(ident_indizes)

            return_dict = dict()
            for key in self.adversarial_samples:
                #print('vs', self.adversarial_samples[key])
                return_dict[key] = len([i for i in self.adversarial_samples[key] if i in ident_indizes])

            return return_dict

    def get_sample_sents(self, typ, predicted, amount, datahandler):
        indizes = self._get_data_ids_for_predicted(generation_type)

        interested_samples = []
        predicted_set = set(self.adversarial_samples[predicted])
        for ident_idx, sample_idx in indizes:
            if ident_idx in predicted_set:
                interested_samples.append(sample_idx)
            if len(interested_samples) == amount:
                break

        return self.adv_sample_handler.get_adversarial_samples_for(interested_samples, datahandler)


    def natural_prediction_dict(self):
        return self.natural_sample_count

    def cnt_natural_samples(self):
        return sum([len(self.adv_sample_handler.samples[gold_label]) for gold_label in self.adv_sample_handler.valid_labels])

    def cnt_w1(self):
        return self.adv_sample_handler.cnt_w1

    def cnt_w2(self):
        return self.adv_sample_handler.cnt_w2

    def accuracy_natural(self):
        return evaluate.accuracy_prediction_dict(self.natural_sample_count)

    def recall_prec_natural(self, label):
        return evaluate.recall_precision_prediction_dict(self.natural_sample_count, label)

    def recall_prec_adversarial(self):
        prediction_dict = dict()
        prediction_dict[self.adv_sample_handler.label] = self.adversarial_sample_count
        #print(prediction_dict)
        return evaluate.recall_precision_prediction_dict(prediction_dict, self.adv_sample_handler.label)

    def accuracy_adversarial(self):
        prediction_dict = dict()
        prediction_dict[self.adv_sample_handler.label] = self.adversarial_sample_count
        return evaluate.accuracy_prediction_dict(prediction_dict)

    def cnt_adversarial(self, generation_type):
        if generation_type not in self.generation_types:
            print('must have on of the following generation types', self.generation_types)
            1/0

        cnt1 = len(self.adv_sample_handler.adversarial_samples_w2_premise)
        cnt2 = len(self.adv_sample_handler.adversarial_samples_w2_hyp)
        cnt3 = len(self.adv_sample_handler.adversarial_samples_w1_premise)
        cnt4 = len(self.adv_sample_handler.adversarial_samples_w1_hyp)

        if generation_type == self.TYPE_SWAP_ANY:
            return cnt1 + cnt2 + cnt3 + cnt4
        elif generation_type == self.TYPE_SWAP_W1_TO_W2:
            return cnt3 + cnt4
        else:
            return cnt1 + cnt2

    def valid_labels(self):
        return self.adv_sample_handler.valid_labels

    def adversarial_label(self):
        return self.adv_sample_handler.label

    def save(self, directory, filename):
        '''
        Save the evaluations at the given file
        '''
        if not self.evaluated:
            print('Must evaluate before saving!')
            1/0

        if filename.split('.')[-1] != 'wpair':
            filename += '.wpair'

        # write to file
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(os.path.join(directory, filename), 'w') as f_out:
            f_out.write(self.adv_sample_handler.w1 + ' ' + self.adv_sample_handler.w2 + '\n')
            f_out.write(self.adv_sample_handler.label + '\n')
            f_out.write(stringify(self.adv_sample_handler.valid_labels) + '\n')

            # now add natural samples
            for label in self.adv_sample_handler.valid_labels:
                f_out.write(stringify(self.adv_sample_handler.samples[label]) + '\n')

            # now adversarial samples
            f_out.write(stringify(self.adv_sample_handler.adversarial_samples_w1_premise) + '\n')
            f_out.write(stringify(self.adv_sample_handler.adversarial_samples_w1_hyp) + '\n')
            f_out.write(stringify(self.adv_sample_handler.adversarial_samples_w2_premise) + '\n')
            f_out.write(stringify(self.adv_sample_handler.adversarial_samples_w2_hyp) + '\n')

            # individual word representative counts
            f_out.write(str(self.adv_sample_handler.cnt_w1) + ' ' + str(self.adv_sample_handler.cnt_w2) + '\n')

            # now evaluation results natural samples
            for gold_label in self.adv_sample_handler.valid_labels:
                f_out.write(stringify([self.natural_sample_count[gold_label][predicted] for predicted in self.adv_sample_handler.valid_labels]) + '\n')
            for gold_label in self.adv_sample_handler.valid_labels:
                for predicted in self.adv_sample_handler.valid_labels:
                    f_out.write(stringify(self.natural_samples[gold_label][predicted]) + '\n')

            # now evaluation results of adversarial samples
            f_out.write(stringify([self.adversarial_sample_count[predicted] for predicted in self.adv_sample_handler.valid_labels]) + '\n')
            for predicted in self.adv_sample_handler.valid_labels:
                f_out.write(stringify(self.adversarial_samples[predicted]) + '\n')


    def _load(self, lines):
        #for i, line in enumerate(lines):
        #    print(i, ':', len(line.split(' ')))
        # general stats
        splitted1 = lines[0].split(' ')
        w1 = splitted1[0]
        w2 = splitted1[1]
        label = lines[1]
        valid_labels = lines[2].split(' ')

        ext_res_pair = data_tools.ExtResPairData(w1, w2, label, valid_labels)

        # natural samples:
        current_line = 3
        ext_res_pair.samples = dict([(lbl, []) for lbl in valid_labels])
        for lbl in valid_labels:
            ext_res_pair.samples[lbl] = intify(lines[current_line])
            current_line += 1
        

        # adversarial samples
        
        ext_res_pair.adversarial_samples_w1_premise = intify(lines[current_line])#[int(idx) for idx in lines[current_line].split(' ')]
        ext_res_pair.adversarial_samples_w1_hyp = intify(lines[current_line + 1])#[int(idx) for idx in lines[current_line + 1].split(' ')]
        ext_res_pair.adversarial_samples_w2_premise = intify(lines[current_line + 2])#[int(idx) for idx in lines[current_line + 2].split(' ')]
        ext_res_pair.adversarial_samples_w2_hyp = intify(lines[current_line + 3])#[int(idx) for idx in lines[current_line + 3].split(' ')]

        # counts
        current_line = current_line + 4
        splitted = lines[current_line].split()
        ext_res_pair.cnt_w1 = int(splitted[0])
        ext_res_pair.cnt_w2 = int(splitted[1])

        # evaluation natural
        self.natural_sample_count = dict([(lbl, dict([(lbl2, 0) for lbl2 in valid_labels])) for lbl in valid_labels])
        self.natural_samples = dict([(lbl, dict([(lbl2, []) for lbl2 in valid_labels])) for lbl in valid_labels])

        current_line += 1
        for gold_label in valid_labels:
            splitted = [int(v) for v in lines[current_line].split(' ')]
            for j, predicted_label in enumerate(valid_labels):
                self.natural_sample_count[gold_label][predicted_label] = splitted[j]
            current_line += 1

        for gold_label in valid_labels:
            for predicted_label in valid_labels:
                indizes = intify(lines[current_line])#[int(v) for v in lines[current_line].split(' ')]
                self.natural_samples[gold_label][predicted_label] = [ext_res_pair.samples[gold_label][idx] for idx in indizes]
                current_line += 1
        
        # evaluation adversarial samples
        splitted = intify(lines[current_line])#[int(v) for v in lines[current_line].split(' ')]
        self.adversarial_sample_count = dict([(valid_labels[i], splitted[i]) for i in range(len(splitted))])
        current_line += 1

        self.adversarial_samples = dict()
        for lbl in valid_labels:
            indizes = intify(lines[current_line])#[int(i) for i in lines[current_line].split(' ')]
            self.adversarial_samples[lbl] = indizes
            current_line += 1

        self.adv_sample_handler = ext_res_pair




