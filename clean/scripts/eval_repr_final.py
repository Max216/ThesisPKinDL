import sys, os, json
sys.path.append('./../')

from libs import model_tools, compatability, evaluate, data_handler, embeddingholder, collatebatch, data_tools
from docopt import docopt

import torch
from torch.utils.data import DataLoader
import torch.autograd as autograd

def main():
    args = docopt("""

    Usage:
        eval_repr_final.py store_repr <model_path> <dataset_path> <out_path>
    """)

    if args['store_repr']:
        store_repr(args['<model_path>'], args['<dataset_path>'], args['<out_path>'])


def store_repr(model_path, data_path, out_path):
    print('Load embeddings')
    embedding_holder = embeddingholder.create_embeddingholder()
    print('Load data')
    dataholder = data_handler.Datahandler(data_path, data_format='snli', include_start_end_token=True)
    print('Load model')
    classifier_name, classifier, embedding_holder = model_tools.load(model_path, embedding_holder)
    print('Predict')

    index_to_tag = data_tools.DEFAULT_VALID_LABELS
    with open(out_path, 'w') as f_out:
        data_loader = DataLoader(dataholder.get_dataset(embedding_holder), drop_last=False, batch_size=1, shuffle=False, collate_fn=collatebatch.CollateBatch(embedding_holder.padding()))
        for premise_batch, hyp_batch, lbl_batch in data_loader:
            prediction, _, representations = classifier(
                autograd.Variable(premise_batch),
                autograd.Variable(hyp_batch),
                output_sent_info = True
            )

            _, predicted_idx = torch.max(scores.data, dim=1)
            pred_label = index_to_tag[predicted_idx.data[0]]
            gold_label = index_to_tag[lbl_batch.data[0]]

            premise_repr = representations[0][0].data
            hyp_repr = representations[1][0].data

            print('pred:', pred_label)
            print('gold:', gold_label)
            print('premise_repr', premise_repr)
            print('hyp_repr', hyp_repr)



if __name__ == '__main__':
    main()