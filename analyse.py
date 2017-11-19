import torch
import torch.autograd as autograd
from torch.utils.data import DataLoader

import train
import model
from model import cuda_wrap


sent_repr_dimen = 256

def idx_to_w(idxs, word_dict):
	return [word_dict[i] for i in idxs.data.numpy()]


def analyse(model, data, batch_size, embedding_holder):
    loader = DataLoader(data, 
                        drop_last = False,    # drops last batch if it is incomplete
                        batch_size=batch_size, 
                        shuffle=False, 
                        #num_workers=0, 
                        collate_fn=train.CollocateBatch(embedding_holder.padding()))

    reverse_dict = embedding_holder.reverse()

    for i_batch, (batch_p, batch_h, batch_lbl) in enumerate(loader):
        premises = autograd.Variable(cuda_wrap(batch_p))
        hypothesis = autograd.Variable(cuda_wrap(batch_h))
        predictions, indizes = model(premises, hypothesis, output_sent_info=True)

        predictions = predictions.data

        indizes_p = indizes[0].data
        indizes_h = indizes[1].data

        lbls = batch_lbl.numpy()

        for i in range(premises.size()[1]):
        	w_premise = idx_to_w(premises[:,i], reverse_dict)
        	w_hypothesis = idx_to_w(hypothesis[:,i], reverse_dict)

        	 
        	print('premise:', )
        	print('hypothesis:', )
        	print('predictions:', predictions.numpy()[i], 'gold:', lbls[i])

        print('predictions', predictions)
        print('indizes', indizes_p, indizes_h)

