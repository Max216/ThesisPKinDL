import sys, os
sys.path.append('./../')

from docopt import docopt

from libs import model_tools, data_tools
from libs import compatability, evaluate
from libs import adversarial_evaluator as adve

def float_string(val):
    return str(round(val, 2))

def flatten(arr):
    flat_list = []
    for sublist in arr:
        for item in sublist:
            flat_list.append(item)
    return flat_list

def summary(file_path, out_path):
    with open(file_path) as f_in:
        lines = [line.strip() for line in f_in.readlines()]

    directory = os.path.dirname(os.path.realpath(file_path))

    header = [
        'review, premise-word (w1)', 'hypothesis-word (w2)', 'assumed label', 
        'generation','# adversarials', 'recall/acc', '# pred. entailment', '# pred. neutral', '# pred. contradiction', 
        '# sents w/ w1', '# sents w/ w2',
        '# natural samples', '# entailment', '# neutral', '# contradiction', 
        'accuracy', 'prec (entailment)', 'recall (entailment)', 'prec (neutral)', 'recall (neutral)', 'prec (contradiction)', 'recall (contradiction)'
    ]

    out_path_examples = out_path.split('.')
    out_path_examples[-1] = '.exmpls'
    out_path_examples = '.'.join(out_path_examples)

    datahandler = data_tools.get_datahandler_train()

    with open(out_path_examples, 'w') as samples_out:
        with open(out_path, 'w') as f_out:

            f_out.write(','.join(header) + '\n')

            for line in lines:
                items = line.split(' ')
                if len(items) != 17:
                    print('Something is wrong', len(items))
                    1/0

                w1 = items[0]
                w2 = items[1]
                rel_path = items[2]

                # Load this for additional information
                c_evaluator = adve.AdvEvaluator(path=os.path.join(directory, rel_path))
                assumed_label = c_evaluator.adversarial_label()

                cnt_natural_samples = items[3]
                cnt_w1_in_sents = items[4]
                cnt_w2_in_sents = items[5]

                # evaluation (is wrong in summary, at least in first versions -> recalc)

                valid_labels = c_evaluator.valid_labels()

                natural_prediction_dict = c_evaluator.natural_prediction_dict()
                natural_sample_counts = dict()
                for gold_label in natural_prediction_dict:
                    natural_sample_counts[gold_label] = sum([natural_prediction_dict[gold_label][predicted] for predicted in natural_prediction_dict[gold_label]])
                
                cnt_natural_entailment = str(natural_sample_counts['entailment'])
                cnt_natural_neutral  = str(natural_sample_counts['neutral'])
                cnt_natural_contradiction = str(natural_sample_counts['contradiction'])


                accuracy_natural = float_string(c_evaluator.accuracy_natural())
                rec_natural_ent, prec_natural_ent = c_evaluator.recall_prec_natural(label='entailment')
                rec_natural_neutral, prec_natural_neutral = c_evaluator.recall_prec_natural(label='neutral')
                rec_natural_contr, prec_natural_contr = c_evaluator.recall_prec_natural(label='contradiction')


                # adversarial evaluation for all generation methods
                generation_types = c_evaluator.generation_types

                samples_out.write('# ' + w1 + ' - ' + w2 + '\n')
                samples_out.write('## Natural samples\n')
                for gold in valid_labels:
                    for predicted_label in valid_labels:
                        samples_out.write('### gold:' + gold + ', predicted:' + predicted_label + '\n')
                        c_evaluator.get_natural_samples(gold, predicted_label, 5, datahandler)

                for typ in generation_types:

                    cnt_adversarial_samples = str(c_evaluator.cnt_adversarial(typ))
                    
                    adv_prediction_dict = c_evaluator.adversarial_prediction_dict(typ)
                    eval_dict = dict()
                    eval_dict[assumed_label] = adv_prediction_dict
                    #print(eval_dict)
                    adv_recall, adv_precision = evaluate.recall_precision_prediction_dict(eval_dict, assumed_label)
                    
                    cnt_adv_predicted_entailment = str(adv_prediction_dict['entailment'])
                    cnt_adv_predicted_neutral = str(adv_prediction_dict['neutral'])
                    cnt_adv_predicted_contradiction = str(adv_prediction_dict['contradiction'])

                    line = ','.join([
                        'unchecked', w1, w2, assumed_label, typ, cnt_adversarial_samples, 
                        float_string(adv_recall), cnt_adv_predicted_entailment, cnt_adv_predicted_neutral, cnt_adv_predicted_contradiction,
                        cnt_w1_in_sents, cnt_w2_in_sents,
                        cnt_natural_samples, cnt_natural_entailment, cnt_natural_neutral, cnt_natural_contradiction,
                        accuracy_natural, 
                        float_string(prec_natural_ent), float_string(rec_natural_ent),
                        float_string(prec_natural_neutral), float_string(rec_natural_neutral),
                        float_string(prec_natural_contr), float_string(rec_natural_contr)
                    ])
                    print(line)
                    f_out.write(line + '\n')

                    # now get sentences
                    samples_out.write('##' + typ + '\n')
                    sample_sents = c_evaluator.get_sample_sents(typ, predicted='entailment', amount=5, datahandler=datahandler)
                    for p, h in sample_sents:
                        samples_out.write('[p] ' + ' '.join(p) + '\n')
                        samples_out.write('[h] ' + ' '.join(h) + '\n')




def write_evaluator_summary(path, evaluator_list):
    with open(path,'w') as f_out:
        for filename, ev in evaluator_list:
            # write all info in one line
            general_info = [ev.adv_sample_handler.w1, ev.adv_sample_handler.w2, filename]
            count_info = [ev.cnt_natural_samples(), ev.cnt_w1(), ev.cnt_w2(), ev.cnt_adversarial(ev.TYPE_SWAP_ANY)]

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
        eval_replacements.py analyse <file>
        eval_replacements.py summary <file> <outpath>

    """)

    path_model = args['<model>']
    swap_words = args['<swap_word_file>']
    exp_name = args['<experiment_name>']
    analyse_file = args['<file>']

    # load model
    if path_model != None and path_model.split('.')[-1] == 'model':
        # use compatability and older model
        print('Use compatability...')
        compa = compatability.Compatablility()
        classifier_name, classifier, embedding_holder = compa.load_model(path_model)
        datahandler_train = compa.get_datahandler_train()

    elif path_model != None:
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

    if args['analyse']:
        evaluator = adve.AdvEvaluator(path=analyse_file)
    elif args['summary']:
        result_path = args['<outpath>']
        if result_path.split('.')[-1] != 'csv':
            result_path += '.csv'
        summary(analyse_file, result_path)
        

if __name__ == '__main__':
    main()