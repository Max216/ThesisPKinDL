import sys, os
sys.path.append('./../')

from docopt import docopt

from libs import model_tools, compatability, evaluate, data_handler, embeddingholder
from exportable import adv_dataset

def main():
    args = docopt("""Evaluate a model on an adversarial dataset.

    Usage:
        evaluate_adversarial_samples.py evaluate <model_path> <dataset_path>
        evaluate_adversarial_samples.py experiment1 <model_path> <dataset_path> <output_path>
    """)

    model_path = args['<model_path>']
    dataset_path = args['<dataset_path>']
    output_path = args['<output_path>']

    # load model
    if model_path.split('.')[-1] == 'model':
        # use compatability and older model
        print('Use compatability...')
        compa = compatability.Compatablility()
        classifier_name, classifier, embedding_holder = compa.load_model(model_path)

    else:
        # use normal model
        classifier_name, classifier, embedding_holder = model_tools.load(model_path)

    embedding_holder = embeddingholder.create_embeddingholder()
    dataholder = data_handler.Datahandler(dataset_path, data_format='snli_adversarial')
    if args['evaluate']:
        
        categories = dataholder.get_categories()
        for category in categories:
            dataset = dataholder.get_dataset_for_category(embedding_holder, category)
            accuracy = evaluate.eval(classifier, dataset, 32, embedding_holder.padding())
            print('Accuracy on', category, '->', accuracy)

        print('Accuracy over all data ->', evaluate.eval(classifier, dataholder.get_dataset(embedding_holder), 8, embedding_holder.padding()))
    else:
        total_correct = 0
        total_amount = 0
        not_count_categories = ['colors', 'at-verbs', 'instruments', 'fruits', 'synonyms', 'fastfood']
        with open(output_path, 'w') as f_out:
            for category in dataholder.get_categories():
                data = dataholder.get_samples_for_category(category)
                f_out.write('\n#', category, 'incorrect:\n')
                category_total = 0
                category_correct = 0
                for p,h,lbl in data:
                    prediction = evaluate.predict_tokenized(classifier, embedding_holder, p, h)
                    if prediction != lbl:
                        f_out.write('[p]',p,'\n')
                        f_out.write('[h]',h, '\n')
                        f_out.write('assumed:', lbl, '; predicted:', prediction, '\n\n')
                    else:
                        if category not in not_count_categories:
                            total_correct += 1
                        category_correct +=1
                    if category not in not_count_categories:
                        total_amount += 1
                    category_total += 1

                print('Acuracy on', category, category_correct / category_total)

            print('Accuracy without:', ','.join(not_count_categories),':')
            print('amount samples:', total_amount, ' ; accuracy:', total_correct / total_amount)

    #def prediction_fn(samples, wp_path):
    #    dataholder = data_handler.Datahandler(wp_path)
    #    return evaluate.predict_outcomes(classifier, dataholder.get_dataset(embedding_holder), 32, embedding_holder.padding())
    #    #return [evaluate.predict_untokenized(classifier, embedding_holder, p, h) for p,h in samples]


    #adv_dataset.evaluate(prediction_fn, dataset_path, output_path,print_samples=5)


if __name__ == '__main__':
    main()