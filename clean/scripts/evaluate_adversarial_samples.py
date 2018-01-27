import sys, os
sys.path.append('./../')

from docopt import docopt

from libs import model_tools, compatability, evaluate
from exportable import adv_dataset

def main():
    args = docopt("""Evaluate a model on an adversarial dataset.

    Usage:
        evaluate_adversarial_samples.py <model_path> <dataset_path> <output_path>
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

    def prediction_fn(samples, wp_path):
        return [evaluate.predict_untokenized(classifier, embedding_holder, p, h) for p,h in samples]


    adv_dataset.evaluate(prediction_fn, dataset_path, output_path,print_samples=5)


if __name__ == '__main__':
    main()