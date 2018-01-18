import sys, os
sys.path.append('./../')

from docopt import docopt

from libs import model_tools, data_tools
from libs import compatability
from libs import adversarial_evaluator as adve

def flatten(arr):
    flat_list = []
    for sublist in arr:
        for item in sublist:
            flat_list.append(item)
    return flat_list

def write_evaluator_summary(path, evaluator_list):
    with open(path,'w') as f_out:
        for filename, ev in evaluator_list:
            # write all info in one line
            general_info = [ev.adv_sample_handler.w1, ev.adv_sample_handler.w2, filename]
            count_info = [ev.cnt_natural_samples(), ev.cnt_w1(), ev.cnt_w2(), ev.cnt_adversarial()]

            # evaluation
            labels = ev.adv_sample_handler.valid_labels

            # natural
            eval_natural = [ev.accuracy_natural()] + flatten([ev.recall_prec_natural(lbl) for lbl in labels])

            # adversarial
            eval_adversarial = [ev.accuracy_adversarial()] + flatten([ev.recall_prec_adversarial()])

            # together
            line = general_info + count_info + eval_natural + eval_adversarial
            line = ' '.join([str(v) for v in line])
            f_out.write(line + '\n')

def main():
    args = docopt("""Evaluate how good a model performs with adversarial examples created by swapping words.

    Usage:
        eval_replacements.py file <model> <swap_word_file> <experiment_name>

    """)

    path_model = args['<model>']
    swap_words = args['<swap_word_file>']
    exp_name = args['<experiment_name>']

    # load model
    if path_model.split('.')[-1] == 'model':
        # use compatability and older model
        print('Use compatability...')
        compa = compatability.Compatablility()
        classifier_name, classifier, embedding_holder = compa.load_model(path_model)
        datahandler_train = compa.get_datahandler_train()

    else:
        # use normal model
        classifier_name, classifier, embedding_holder = model_tools.load(path_model)
        datahandler_train = data_tools.get_datahandler_train()


    if args['file']:
        res_handler = data_tools.ExtResPairhandler(swap_words)
        all_pair_data = res_handler.find_samples(datahandler_train)

        evaluators = []
        directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../results/analysis/swap_words/' + exp_name + '/' + classifier_name)
        
        for pair_data in all_pair_data:
            evaluator = adve.AdvEvaluator()
            filename = pair_data.w1 + '_' + pair_data.w2 + '.wpair'
            evaluator.evaluate(classifier, pair_data, datahandler_train, embedding_holder)
            evaluator.save(directory, filename)
            print('Saved', filename)
            evaluators.append((filename, evaluator))

        summary_path = os.path.join(directory, 'summary.stxt')
        write_evaluator_summary(summary_path, evaluators)
        

if __name__ == '__main__':
    main()