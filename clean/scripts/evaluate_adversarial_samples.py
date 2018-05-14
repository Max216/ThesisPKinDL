import sys, os, json
sys.path.append('./../')

from docopt import docopt

from libs import model_tools, compatability, evaluate, data_handler, embeddingholder
from exportable import adv_dataset

def main():
    args = docopt("""Evaluate a model on an adversarial dataset.

    Usage:
        evaluate_adversarial_samples.py evaluate <model_path> <dataset_path> [--embd1=<embd1>] [--embd2=<embd2>]
        evaluate_adversarial_samples.py experiment1 <model_path> <dataset_path> <output_path>
        evaluate_adversarial_samples.py eo <model_path> <dataset_path> <out_folder> [--embd1=<embd1>] [--embd2=<embd2>]
    """)

    model_path = args['<model_path>']
    dataset_path = args['<dataset_path>']
    output_path = args['<output_path>']

    embd1 = args['--embd1']
    embd2 = args['--embd2']

    embedding_holder = embeddingholder.create_embeddingholder()
    if embd1 != None:
        embedding_holder.concat(embeddingholder.EmbeddingHolder(embd1))
    if embd2 != None:
        embedding_holder.concat(embeddingholder.EmbeddingHolder(embd2))

    # load model
    if model_path.split('.')[-1] == 'model':
        # use compatability and older model
        print('Use compatability...')
        compa = compatability.Compatablility()
        classifier_name, classifier, embedding_holder = compa.load_model(model_path)

    else:
        # use normal model
        classifier_name, classifier, embedding_holder = model_tools.load(model_path, embedding_holder)

    

    dataholder = data_handler.Datahandler(dataset_path, data_format='snli_adversarial')
    if args['evaluate']:
        categories = dataholder.get_categories()
        for category in categories:
            dataset = dataholder.get_dataset_for_category(embedding_holder, category)
            accuracy = evaluate.eval(classifier, dataset, 1, embedding_holder.padding())
            print('Accuracy on', category, '->', accuracy)

        print('Accuracy over all data ->', evaluate.eval(classifier, dataholder.get_dataset(embedding_holder), 1, embedding_holder.padding()))
        


    elif args['eo']:
        print('Read raw data')
        with open(dataset_path) as f_in:
            raw_data = [json.loads(line.strip()) for line in f_in.readlines()]

        dataset = data_handler.get_datahandler_dev(path=dataset_path).get_dataset(embedding_holder)
        outcomes, golds = evaluate.predict_outcomes2(classifier, dataholder.get_dataset(embedding_holder), 1, embedding_holder.padding())
        #print('Accuracy over all data ->', evaluate.eval(classifier, dataholder.get_dataset(embedding_holder), 1, embedding_holder.padding()))
    
        appendix = dataset_path.split('.')[-1]
        outpath = os.path.join(args['<out_folder>'], classifier_name + '.' + appendix)

        if not os.path.exists(args['<out_folder>']):
            os.makedirs(args['<out_folder>'])

        with open(outpath, 'w') as f_out:
            for i in range(len(outcomes)):
                gold = golds[i]
                pred = outcomes[i]
                data_gold = raw_data[i]['gold_label']
                sample = raw_data[i]

                # verify
                if data_gold != gold:
                    print('nope:', data_gold, 'vs', gold)
                    1/0

                sample['predicted_label'] = pred
                f_out.write(json.dumps(sample) + '\n')

        

        
    else:
        total_correct = 0
        total_amount = 0
        not_count_categories = []
        with open(output_path, 'w') as f_out:
            for category in dataholder.get_categories():
                data = dataholder.get_samples_for_category(category)
                f_out.write('\n# ' + category + ' incorrect:\n')
                category_total = 0
                category_correct = 0
                for p,h,lbl in data:
                    prediction = evaluate.predict_tokenized(classifier, embedding_holder, p, h)
                    if prediction != lbl:
                        f_out.write('[p] ' + ' '.join(p) + '\n')
                        f_out.write('[h] ' + ' '.join(h) + '\n')
                        f_out.write('assumed: ' +  lbl + '; predicted: ' + prediction + '\n\n')
                    else:
                        if category not in not_count_categories:
                            total_correct += 1
                        category_correct +=1
                    if category not in not_count_categories:
                        total_amount += 1
                    category_total += 1

                print('Acuracy on', category, category_correct / category_total)

            print('Accuracy without: ' +  ','.join(not_count_categories) + ':')
            print('amount samples: ' + str(total_amount) + '; accuracy: ' + str(total_correct / total_amount))

    #def prediction_fn(samples, wp_path):
    #    dataholder = data_handler.Datahandler(wp_path)
    #    return evaluate.predict_outcomes(classifier, dataholder.get_dataset(embedding_holder), 32, embedding_holder.padding())
    #    #return [evaluate.predict_untokenized(classifier, embedding_holder, p, h) for p,h in samples]


    #adv_dataset.evaluate(prediction_fn, dataset_path, output_path,print_samples=5)


if __name__ == '__main__':
    main()