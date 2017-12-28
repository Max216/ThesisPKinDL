'''
This file includes model and methods to train and load data where each sentence representation
(of premise, hypothesis) is threat individually, thus not using them as a pair.
'''

import model as m

from docopt import docopt
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda as cu
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.cuda as cu

import numpy as np
import random
import copy

import sys

def chunker(seq, size):
	return (seq[pos:pos + size] for pos in range(0, len(seq), size))

class ReprClassifierNoPair(nn.Module):
	'''
	This NN uses pretrained sentence representations to classify some labels, but does not use feature
	concatenation and uses premise, hypothesis individuallly.
	'''

	def __init__(self, input_dim, hidden_dim, output_dim, dropout):
		super(ReprClassifierNoPair, self).__init__()

		self.hidden1 = nn.Linear(input_dim, hidden_dim)
		self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
		self.hidden3 = nn.Linear(hidden_dim, output_dim)
		self.dropout1 = nn.Dropout(p=dropout)
		self.dropout2 = nn.Dropout(p=dropout)

	def forward(self, rep):
		out = self.dropout1(F.relu(self.hidden1(rep)))
		out = self.dropout2(F.relu(self.hidden2(out)))
		return F.softmax(self.hidden3(out))

def evaluate(classifier, data_set):
	classifier.eval()

	correct = 0
	total = 0

	for rep, lbls in data_set:
		total += rep.size()[0]

		var_rep = m.cuda_wrap(autograd.Variable(rep))
		var_lbl = m.cuda_wrap(autograd.Variable(lbls))

		predictions = classifier(var_rep)
		_, predicted_idx = torch.max(predictions, dim=1)

		correct += torch.sum(torch.eq(var_lbl, predicted_idx)).data[0]

	classifier.train()

	return correct / total

def store_best_result(filename, classifier, data_train, data_dev, labels):
	classifier.eval()

	lines = []
	
	for name, dataset in [('train', data_train), ('dev', data_dev)]:
		# create misclassification dict
		misclassification_dict = dict()
		for lbl in labels:
			misclassification_dict[lbl] = dict()
			for lbl2 in labels:
				misclassification_dict[lbl][lbl2] = 0

		for batch_rep, batch_lbl in dataset:
			var_rep = m.cuda_wrap(autograd.Variable(batch_rep))
			var_lbl = m.cuda_wrap(autograd.Variable(batch_lbl))

			predictions = classifier(var_rep)
			_, predicted_idx = torch.max(predictions, dim=1)
			
			gold_labels = var_lbl.data
			predicted_labels = predicted_idx.data
			for i in range(gold_labels.size()[0]):
				gold = labels[gold_labels[i]]
				predicted = labels[predicted_labels[i]]
				misclassification_dict[gold][predicted] += 1

		for lbl_gold in labels:
			for lbl_predicted in labels:
				lines.append('-'.join([name, str(lbl_gold), str(lbl_predicted), str(lbl_predicted)]) + ' ' + str(misclassification_dict[lbl_gold][lbl_predicted]) + '\n')

	with open('./analyses/' + filename, 'w') as f_out:
		for line in lines:
			f_out.write(line)

	print('Done writing files.')


def train(classifier, dataset_train, dataset_train_eval, dataset_dev, iterations, lr, validate_after=2000):
	classifier = m.cuda_wrap(classifier)
	classifier.train()

	until_validation = 0
	samples_seen = 0
	best_dev_acc = -1
	best_train_acc = -1
	best_model = None
	optimizer = optim.Adam(classifier.parameters(), lr=lr)

	for i in range(iterations):
		print('train iteration', i+1)

		for batch_rep, batch_lbl in dataset_train:

			until_validation -= batch_rep.size()[0]
			samples_seen += batch_rep.size()[0]

			# undo previous gradients
			classifier.zero_grad()
			optimizer.zero_grad()

			var_sents = autograd.Variable(m.cuda_wrap(batch_rep))
			var_lbls = autograd.Variable(m.cuda_wrap(batch_lbl))

			prediction = classifier(var_sents)
			mean_loss = F.cross_entropy(prediction, var_lbls)

			mean_loss.backward()
			optimizer.step()

			if until_validation <= 0:
				until_validation = validate_after
				print('After seeing', samples_seen, 'samples:')
				train_acc = evaluate(classifier, dataset_train_eval)
				dev_acc = evaluate(classifier, dataset_dev)
				print('Acc-train', train_acc)
				print('Acc-dev', dev_acc)
				

				if dev_acc > best_dev_acc:
					best_dev_acc = dev_acc
					best_train_acc = train_acc
					print('Current best!')
					best_model = copy.deepcopy(classifier.state_dict())

				sys.stdout.flush()

	print('Done. Best Accuracy: train:', best_train_acc, 'dev:', best_dev_acc)
	return best_model


def find_count_labels(folder):
	all_labels = set()

	for data_type in ['train', 'dev']:
		pre = folder + 'invert_4m4f_' + data_type
		file_names = [
			pre + '-correct_correct.txt',
			pre + '-correct_incorrect.txt',
			pre + '-incorrect_correct.txt',
			pre + '-incorrect_incorrect.txt'
		]

		for file in file_names:
			with open(file) as f_in:
				for data in chunker(f_in.readlines(), 9):
					for idx in [0, 3]:
						all_labels.add(len(data[idx].strip().split(' ')))

	idx_to_labels = list(all_labels)
	return idx_to_labels, dict([(lbl, i) for i, lbl in enumerate(idx_to_labels)]) 



def load_data_from_folder(folder, data_type, batch_size, label_fn, shuffle=True):
	'''
	Load all data from a folder with classified/misclassified samples.

	:param folder 	Where the files are stored
	:param data_type 	either "train" or "dev"
	'''



	pre = folder + 'invert_4m4f_' + data_type
	file_names = [
		pre + '-correct_correct.txt',
		pre + '-correct_incorrect.txt',
		pre + '-incorrect_correct.txt',
		pre + '-incorrect_incorrect.txt'
	]

	all_data = []
	all_hashes = set()

	for file in file_names:
		with open(file) as f_in:
			for data in chunker(f_in.readlines(), 9):
				# check premise
				hash_p = hash(data[0])
				if hash_p not in all_hashes:
					all_hashes.add(hash_p)
					lbl = lbl = label_fn(data[0])
					all_data.append((np.asarray(data[2].strip().split(' '), dtype=float), lbl))

				# check hypothesis
				hash_h = hash(data[3])
				if hash_h not in all_hashes:
					all_hashes.add(hash_h)
					lbl = label_fn(data[3])
					all_data.append((np.asarray(data[5].strip().split(' '), dtype=float), lbl))

	print('Found',len(all_data), 'individual sentences.')
	random.shuffle(all_data)
	all_data = [(torch.from_numpy(rep).float(), lbl) for rep, lbl in all_data]
	input_dim = all_data[0][0].shape[0]
	return input_dim, DataLoader(all_data, drop_last=False, batch_size=batch_size, shuffle=True)


def main():
	torch.manual_seed(6)
	args = docopt("""Train a model based on existing sentence representation. Each sentence is used individually, not 
		as a pair <premise, hypothesis>.

	Usage:
    	sent_representation_classify.py sent_len <data_folder>
	""")

	if args['sent_len']:



		data_folder = args['<data_folder>']
		idx_to_lbl, lbl_to_idx = find_count_labels(data_folder)

		def lbl_fn(rep_line):
			return lbl_to_idx[len(rep_line.strip().split(' '))]

		input_dim, data_train = load_data_from_folder(data_folder, 'train', batch_size=32, label_fn=lbl_fn)
		input_dim, data_train_eval = load_data_from_folder(data_folder, 'train', batch_size=32, label_fn=lbl_fn, shuffle=False)
		input_dim, data_dev = load_data_from_folder(data_folder, 'dev', batch_size=32, label_fn=lbl_fn, shuffle=False)

		classifier = ReprClassifierNoPair(input_dim, input_dim // 2, len(idx_to_lbl), 0.0)

		best_classifier = train(classifier, data_train, data_train_eval, data_dev, 5, 0.0002)

		# reload best setting and remeber samples
		classifier.load_state_dict(best_classifier)
		store_best_result('classifications_sent_len.txt', classifier, data_train_eval, data_dev, idx_to_lbl)
		


if __name__ == '__main__':
	main()
