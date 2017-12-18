import model as m
import mydataloader
import train
import embeddingholder
import config
from torch.utils.data import DataLoader
import sys

import torch
import torch.autograd as autograd

from train import CollocateBatchWithSents

from docopt import docopt

def main():
    args = docopt("""Find samples for given gold and predicted label.

    Usage:
        find_samples.py <model> <data> <gold_label> <predicted_label> <lexical_threshold>

        <model> = Path to trained model
        <data>  = Path to data to test model with 
        <gold_label> = onyl look at samples with this gold label
        <predicted_label> only look at samples predicte as this label
        <lexical_threshold> only look at samples that share a lexical overlap based on this
    """)

    find(args['<model>'], args['<data>'], args['<gold_label>'], args['<predicted_label>'], float(args['<lexical_threshold>']))

def find(classifier_path, data_path, gold_label, find_predicted_label, t):
    print('Results for gold label =', gold_label, '; predicted=', find_predicted_label)

    def filter_fn(p, h, lbl):
        if lbl == gold_label:
            abs_overlap = len([w for w in p if w in h])
            rel_overlap = abs_overlap / min([len(p), len(h)])
            if rel_overlap >= t:
                return True
        return False


    embedding_holder = embeddingholder.EmbeddingHolder(config.PATH_WORD_EMBEDDINGS)
    data = mydataloader.get_dataset_chunks(data_path, embedding_holder, filter_fn=filter_fn, include_sent=True)
    classifier, _ = m.load_model(classifier_path, embedding_holder=embedding_holder)
    classifier.eval()
    classifier = m.cuda_wrap(classifier)

    # Check if predicted label is as wanted

    loader = [DataLoader(chunk, drop_last = False, batch_size=1, shuffle=False, collate_fn=CollocateBatchWithSents(embedding_holder.padding())) for chunk in data]

    for chunk in loader:
        for i_batch, (batch_p, batch_h, batch_lbl, sent_p, sent_h) in enumerate(chunk):
            predictions = classifier(autograd.Variable(m.cuda_wrap(batch_p)),
                                autograd.Variable(m.cuda_wrap(batch_h))).data

            _, predicted_idx = torch.max(predictions, dim=1)
            predicted_label = mydataloader.index_to_tag[predicted_idx[0]]
            if predicted_label == find_predicted_label:
                print()
                print('[premise]' + ' '.join(sent_p[0]))
                print('[hypothesis]' + ' '.join(sent_h[0]))
                sys.stdout.flush()
    
    

if __name__ == '__main__':
    main()
