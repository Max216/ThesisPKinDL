'''To evaluate word pairs'''

from libs import evaluate

import os

def stringify(arr):
    return ' '.join([str(v) for v in arr])

class AdvEvaluator:
    '''
    Evaluates adversarial sampels
    '''

    def __init__(self, path=None):
        '''
        Create a new object to gernate the given adversarial samples
        :path       Path where to load information from.
        '''
        self.evaluated = False
        if path != None:
            print('todo')
            1/0
            self.evaluated = True

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
        return evaluate.recall_precision_prediction_dict(prediction_dict, self.adv_sample_handler.label)

    def accuracy_adversarial(self):
        prediction_dict = dict()
        prediction_dict[self.adv_sample_handler.label] = self.adversarial_sample_count
        return evaluate.accuracy_prediction_dict(prediction_dict)

    def cnt_adversarial(self):
        cnt1 = len(self.adv_sample_handler.adversarial_samples_w2_premise)
        cnt2 = len(self.adv_sample_handler.adversarial_samples_w2_hyp)
        cnt3 = len(self.adv_sample_handler.adversarial_samples_w1_premise)
        cnt4 = len(self.adv_sample_handler.adversarial_samples_w1_hyp)
        return cnt1 + cnt2 + cnt3 + cnt4

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

            # now artificial samples
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






