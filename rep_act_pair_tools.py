'''
This tool can be used to analyse predicted data consiting of activations, representations, label, etc
'''

import numpy as np
import torch
import torch.autograd as autograd

from docopt import docopt

import model as m
import embeddingholder
import mydataloader
import config

def chunker(seq, size):
	return (seq[pos:pos + size] for pos in range(0, len(seq), size))

class Sample:
	'''
	A single sample consisting of premise and hypothesis together with 
	all activations and representations is stored compactly in this class.
	'''

	def __init__(self, p, p_act, p_rep, h, h_act, h_rep, lbl, predicted):
		self.p = p
		self.h = h
		self.p_act = p_act
		self.h_act = h_act
		self.p_rep = p_rep
		self.h_rep = h_rep
		self.lbl = lbl
		self.predicted = predicted
		self.dims = [i for i in range(len(p_act))]
		self.applied_filters = []


	def get_grid_items(self, start_p, start_h, bin_size):
		'''
		Get a list of (word-premise repr-premise word-hy repr-hyp, dim-idx) for all items
		fitting into a single grid field.

		:param start_p lower value of the bin for the premise
		:param start_h lower value of the bin for the hypothesis
		:param bin_size  added to the start value to see what items fit
		'''

		return [
			(self.get_premise_word_at_dim(i), self.p_rep[i], self.get_hyp_word_at_dim(i), self.h_rep[i], i) 
			for i in range(len(self.p_rep))
			if  start_p <= self.p_rep[i] < start_p + bin_size and start_h <= self.h_rep[i] < start_h + bin_size
		]


	def get_premise_word_at_dim(self, dim):
		'''
		Return the word of the premise responsible for the value at the given dimension.
		'''
		return self.p[self.p_act[dim]]

	def get_hyp_word_at_dim(self, dim):
		''' 
		Return the word of the hypothesis responsible for the value at the given dimension.
		'''
		return self.h[self.h_act[dim]]

	def filter(self, filter_fn):
		'''
		Filters the data contained in this class using a filter_fn

		:param filter_fn 	must be a function taking a sample as input and returning 
							(name, dims). Only those dimensions in dims are kept.
		'''
		name, keep_dims = filter_fn(self)
		keep_dim_indizes = [i for i in range(len(self.dims)) if self.dims[i] in keep_dims]
		self.dims = np.take(self.dims, keep_dim_indizes)
		self.applied_filters.append(name)

	def get_applied_filters(self):
		'''
		:return the name appendix based on applied filters
		'''
		return '_'.join(self.applied_filters)

	def dimsize(self):
		return len(self.dims)

	def swap(self):
		'''
		swap premise with hypothesis
		'''
		tmp_p = self.p
		tmp_p_act = self.p_act
		tmp_p_rep = self.p_rep

		self.p = self.h
		self.p_act = self.h_act
		self.p_rep = self.h_rep

		self.h = tmp_p
		self.h_act = tmp_p_act
		self.h_rep = tmp_p_rep

def predict_label(classifier, sample):
	'''
	Predict a single sample consisting of of premise and hypothesis into a string output.

	:param classifier 		The model to use for prediction
	:param sample 			The @class Sample with premise and hypothesis
	'''
	embedding_holder = embeddingholder.EmbeddingHolder(config.PATH_WORD_EMBEDDINGS)

	variable_premise = m.cuda_wrap(autograd.Variable(torch.LongTensor([embedding_holder.word_index(w) for w in sample.p]).view (-1, 1)))
	variable_hyp = m.cuda_wrap(autograd.Variable(torch.LongTensor([embedding_holder.word_index(w) for w in sample.h]).view(-1, 1)))

	classifier.eval()
	out = classifier(variable_premise, variable_hyp, output_sent_info=False)
	_, predicted_idx = torch.max(out, dim=1)
	predicted_lbl = mydataloader.index_to_tag[predicted_idx.data[0]]
	
	return predicted_lbl

def load_correct_sentences(path=None):
	'''
	Only loads correct samples.

	:param path 	Path to the file containing the information
	'''
	SAMPLE_SIZE = 7
	DEFAULT_PATH = './analyses/representation_samples_450_150_150_150.txt'

	path = path or DEFAULT_PATH

	def to_sample(data_chunk):
		p = data_chunk[0].strip().split(' ')
		p_act = np.asarray(data_chunk[1].strip().split(' '), dtype=int)
		p_rep = np.asarray(data_chunk[2].strip().split(' '), dtype=float)
		h = data_chunk[3].strip().split(' ')
		h_act = np.asarray(data_chunk[4].strip().split(' '), dtype=int)
		h_rep = np.asarray(data_chunk[5].strip().split(' '), dtype=float)
		lbl_gold = data_chunk[6].strip().split(' ')[0]
		lbl_predicted = data_chunk[6].strip().split(' ')[0]
		return Sample(p, p_act, p_rep, h, h_act, h_rep, lbl_gold, lbl_predicted)

	with open(path) as f_in:
		return [to_sample(chunk) for chunk in chunker(f_in.readlines()[:150*3*SAMPLE_SIZE], SAMPLE_SIZE)]

def stringify_arr(arr):
	return ' '.join([str(v) for v in arr]) + '\n'

def swap_predict(classifier, data, result_path):
	'''
	Predict a sample but swap hypothesis with premise. Samples are printed together with the original and swapped
	prediction. Quantitative results are stored in a file with the formal <gold_label>-<predicted>-<swap_predicted> <amount>.

	:param classifier 	classifier to use
	:param data 		list of samples
	:param result_path 	filename for quantitative results
	'''

	# store quantitative results
	classification_dict = dict()

	# store samples
	samples_dict = dict()

	for lbl_gold in mydataloader.index_to_tag:
		classification_dict[lbl_gold] = dict()
		samples_dict[lbl_gold] = dict()
		for lbl_predicted in mydataloader.index_to_tag:
			classification_dict[lbl_gold][lbl_predicted] = dict()
			samples_dict[lbl_gold][lbl_predicted] = dict()
			for lbl_predicted_swapped in mydataloader.index_to_tag:
				classification_dict[lbl_gold][lbl_predicted][lbl_predicted_swapped] = 0
				samples_dict[lbl_gold][lbl_predicted][lbl_predicted_swapped] = []

	for sample in data:
		# swap 
		sample.swap()

		predicted_swapped = predict_label(classifier, sample)
		classification_dict[sample.lbl][sample.predicted][predicted_swapped] += 1
		samples_dict[sample.lbl][sample.predicted][predicted_swapped].append(sample)
		
	# output
	with open(result_path + '.txt', 'w') as f_out:
		for lbl_gold in classification_dict:
			for lbl_predicted in classification_dict[lbl_gold]:
				if lbl_gold == lbl_predicted:
					for lbl_predicted_swapped in classification_dict[lbl_gold][lbl_predicted]:

						# to file
						f_out.write('-'.join([lbl_gold, lbl_predicted, lbl_predicted_swapped]) + ' ' + str(classification_dict[lbl_gold][lbl_predicted][lbl_predicted_swapped]) + '\n')

						print('# Gold:' + lbl_gold + '; Predicted:' + lbl_predicted + '; Swapped:' + lbl_predicted_swapped)
						with open(result_path + '_' + lbl_predicted + '_' + lbl_predicted_swapped + '.txt', 'w') as f_out2:
							for sample in samples_dict[lbl_gold][lbl_predicted][lbl_predicted_swapped]:
								# to stdout
								print('[p]', ' '.join(sample.p))
								print('[h]', ' '.join(sample.h))
								print()

								# to file
								f_out2.write(stringify_arr(sample.p))
								f_out2.write(stringify_arr(sample.p_act))
								f_out2.write(stringify_arr(sample.p_rep))
								f_out2.write(stringify_arr(sample.h))
								f_out2.write(stringify_arr(sample.h_act))
								f_out2.write(stringify_arr(sample.h_rep))
								f_out2.write(sample.predicted + ' ' + lbl_predicted_swapped + '\n')





def main():
	args = docopt("""Analyse predicted data..

	Usage:
		rep_act_pair_tools.py swapcorrect <model_path> <result_path> [<data_path>]
	""")

	model_path = args['<model_path>']
	result_path = args['<result_path>']
	data_path = args['<data_path>']

	data = load_correct_sentences(data_path)
	classifier, _ = m.cuda_wrap(m.load_model(model_path))
	swap_predict(classifier, data, result_path)

if __name__ == '__main__':
	main()
