'''
This file is only used for further investigation of the male/female dimensions of the trained model.
It is bound to a fixed set of files, if it should be used with other files it jhas to be adopted.
'''


import sys

from docopt import docopt
import numpy as np
import mydataloader
from mydataloader import index_to_tag
from sent_representation_classify import load_simple_model, load_data

import torch
from torch.utils.data import DataLoader
import torch.autograd as autograd
import model as m

import matplotlib.pyplot as plt

import random


FOLDER = './analyses/'
FILE_BASE = 'invert_4m4f_'
FILE_APX_DATA = ['train-', 'dev-']
FILE_CATEGORIES = [
	('Initially and inverted correct', 'correct_correct'),
	('Initially correct and inverted incorrect', 'correct_incorrect'),
	('Initially incorrect and inverted correct', 'incorrect_correct'),
	('Initially and inverted incorrect', 'incorrect_incorrect')
]
FILE_OUT_TWIST_COUNTS = FOLDER + FILE_BASE + 'results.txt'

color_palette = [
'#e6194b', 
'#3cb44b', 
'#ffe119', 
'#0082c8', 
'#f58231', 
'#911eb4', 
'#46f0f0', 
'#000000', 
'#f032e6', 
'#d2f53c', 
'#fabebe', 
'#008080', 
'#aa6e28', 
'#800000', 
'#808000', 
'#000080', 
'#808080', 
'#e6beff',
'#aaffc3',
'#fffac8',
'#ff0000',
'#00ff00',
'#0000ff',
'#ff00ff',
'#234567',
'#00ffff'
]

def init_misclassification_dict(labels):
	d = dict()
	for label in labels:
		d[label] = dict()
		for label2 in labels:
			d[label][label2] = 0
	return d


def plot_correct_incorrect_bar(x_labels, misclassification_dict, title, block=True):
	num_groups = len(x_labels)
	
	# how many classified as label as array for plotting
	amounts = dict()
	for lbl in x_labels:
		amounts[lbl] = [misclassification_dict[l][lbl] for l in x_labels]
	
	# how many of gold label for acc
	amounts_gold = dict()
	for lbl in x_labels:
		amounts_gold[lbl] = sum([misclassification_dict[lbl][l] for l in x_labels])

	# calculate accuracy
	amount_data = [amounts_gold[lbl] for lbl in x_labels]
	correct = [amounts[lbl][i] for i, lbl in enumerate(x_labels)]
	print(title)
	print(amount_data)
	print(correct)
	accuracy = round(sum(correct) / sum(amount_data) * 100, 2)
	lbl_accuracies = [round(correct[i] / amount_data[i] * 100, 2) for i in range(len(x_labels))]

	fig, ax = plt.subplots()
	index = np.arange(num_groups)
	colors = [color_palette[i] for i in index]
	bar_width = .1

	for i, lbl in enumerate(x_labels):
		plt.bar(index + i * bar_width, amounts[lbl], bar_width, color=color_palette[i], label=lbl)

	plt.xlabel('Gold label')
	plt.ylabel('# samples')
	x_labels = [lbl + ' (' + str(lbl_accuracies[i]) + ')' for i, lbl in enumerate(x_labels)]
	plt.xticks(index +  bar_width, x_labels)
	plt.title(title + ' (' + str(accuracy) + ')')
	plt.legend(title='Classified as:', bbox_to_anchor=(0,1.12,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=3)
	plt.subplots_adjust(top=0.8)
	plt.show(block=block)

def plot_diff_inversed_normal(x_labels, correct, inverse_model_dict, title, block=True):
	num_groups = len(x_labels)
	amounts = dict()
	for lbl in x_labels:
		amounts[lbl] = [inverse_model_dict[l][lbl] for l in x_labels]

	fig, ax = plt.subplots()
	index = np.arange(num_groups)
	colors = [color_palette[i] for i in index]
	bar_width = .1

	# plot normal model
	plt.bar(index, correct, bar_width, color=color_palette[num_groups], label='Correct by std model')

	# plot inversed
	for i, lbl in enumerate(x_labels):
		plt.bar(index + (i+1) * bar_width, amounts[lbl], bar_width, color=color_palette[i], label=lbl)

	plt.xlabel('Gold label')
	plt.ylabel('# samples')
	plt.xticks(index +  bar_width * 1.5, x_labels)
	plt.title(title)
	plt.legend(title='Classified as:', bbox_to_anchor=(0,1.12,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=2)
	plt.subplots_adjust(top=0.7)
	plt.show(block=block)

def plot_findings(params):

	if params == None:
		params = 'train'

	with open(FILE_OUT_TWIST_COUNTS) as f_in:
		all_data = [(line.strip().split(' ')[0], int(line.strip().split(' ')[1])) for line in f_in.readlines()]

	filtered_data = [(key.split('-')[1:], amount) for key, amount in all_data if key.split('-')[0] == params]
	filtered_data = [(k[0], k[1], k[2], amount) for k, amount in filtered_data]
	
	x_labels = index_to_tag
	# Plot basic results for normal model
	misclassification_dict_normal = init_misclassification_dict(x_labels)
	for gold, predicted, _, amount in filtered_data:
		misclassification_dict_normal[gold][predicted] += amount

	title = 'Classification (normal model) in ' + params
	plot_correct_incorrect_bar(x_labels, misclassification_dict_normal, title, block=False)

	# plot basic results for inversed model
	misclassification_dict_inv = init_misclassification_dict(x_labels)
	for gold, _, predicted_inv, amount in filtered_data:
		misclassification_dict_inv[gold][predicted_inv] += amount
	title = 'Classification (inversed male/female model) in ' + params
	plot_correct_incorrect_bar(x_labels, misclassification_dict_inv, title, block=False)


	# Plot inversed model compared with nomal model
	correct_normal = [misclassification_dict_normal[lbl][lbl] for lbl in x_labels]
	misclassification_dict_inv_normal = init_misclassification_dict(x_labels)
	for gold, predicted, predicted_inv, amount in filtered_data:
		if gold == predicted:
			misclassification_dict_inv_normal[gold][predicted_inv] += amount
	title = 'M/F inversed where normal model was correct'
	plot_diff_inversed_normal(x_labels, correct_normal, misclassification_dict_inv_normal, title, block=True)

def print_sents(params):

	if params == None:
		num_sents=100
	else:
		num_sents = int(params)

	counter = dict()
	for data_type in FILE_APX_DATA:
		for category_title, category in FILE_CATEGORIES:
			file = FOLDER + FILE_BASE + data_type + category + '.txt'
			print()
			print('Reading:', file)
			print('Showing sentences of', data_type + 'data:', category_title)
			with open(file) as f_in:
				line_idx = 0
				premise = None
				hypothesis = None
				labels = None
				confidences_normal = None

				lines = f_in.readlines()
				size = len(lines) // 9
				random_smpl_indizes = [idx * 9 + 8 for idx in random.sample(range(size), num_sents)]

				for line in lines:
					# check if premise
					if line_idx % 9 == 0:
						premise = line.strip()

					# check if hypothesis
					if line_idx % 9 == 3:
						hypothesis = line.strip()

					# check if label
					if line_idx % 9 == 6:
						labels = [int(v) for v in line.strip().split(' ')]

					if line_idx % 9 == 7:
						confidences_normal = line.strip().split(' ')

					if line_idx % 9 == 8:
						confidences_inv = line.strip().split(' ')
						lbl_gold = index_to_tag[labels[0]]
						lbl_predicted = index_to_tag[labels[1]]
						lbl_predicted_inv = index_to_tag[labels[2]]
						conf_norml = confidences_normal[labels[1]]
						conf_inv = confidences_inv[labels[2]]

						if line_idx in random_smpl_indizes:
							print('premise:', premise)
							print('hypothesis:', hypothesis)
							print('label-gold:', lbl_gold)
							print('label-predicted:', lbl_predicted, '(' + conf_norml + ')')
							print('label-predicted-inv:', lbl_predicted_inv + '(' + conf_inv + ')')
							print('-----')

						# count
						key = data_type + lbl_gold + '-' + lbl_predicted + '-' + lbl_predicted_inv
						if key in counter:
							counter[key] += 1
						else:
							counter[key] = 1


					line_idx += 1

	print('Done.')
	# write out overall statistics for plotting
	print('Write statistics into', FILE_OUT_TWIST_COUNTS)
	with open(FILE_OUT_TWIST_COUNTS, 'w') as f_out:
		for key in counter.keys():
			f_out.write(key + ' ' + str(counter[key]) + '\n')


def plot_stats_at_path(params):
	'''
	plot the accuracies of a model. file must have the following lines:
	<dataset>-<goldlabel>-<predicted label>-<dummy> <amount>

	:params 	like "<path to file> <dataset>"
	'''
	path = params.split(' ')[0]
	dataset = params.split(' ')[1]

	print(path)
	print(dataset)

	with open(path) as f_in:
		all_data = [(line.strip().split(' ')[0], int(line.strip().split(' ')[1])) for line in f_in.readlines()]

	filtered_data = [(key.split('-')[1:], amount) for key, amount in all_data if key.split('-')[0] == dataset]
	filtered_data = [(k[0], k[1], k[2], amount) for k, amount in filtered_data]
	
	x_labels = index_to_tag
	# Plot basic results for normal model
	misclassification_dict_normal = init_misclassification_dict(x_labels)
	for gold, predicted, _, amount in filtered_data:
		misclassification_dict_normal[gold][predicted] += amount

	title = 'Classification (normal model) in ' + dataset
	plot_correct_incorrect_bar(x_labels, misclassification_dict_normal, title)

def print_stats_simple_model(params):
	'''
	prints the statistics needed to plot details of the model

	:params  need to be "path dims=dim1,dim2,dim3"
	'''
	splitted_params = params.split(' ')
	path = splitted_params[0]
	dimensions = [int(d) for d in splitted_params[1].split('=')[1].split(',')]
	input_dim = len(dimensions)
	classifier = load_simple_model(path, input_dim)

	# load data
	data_train = DataLoader(load_data(FOLDER, 'train', dimensions), drop_last=False, batch_size=1, shuffle=False)
	data_dev = DataLoader(load_data(FOLDER, 'dev', dimensions), drop_last=False, batch_size=1, shuffle=False)	
	 

	# go through data
	for name, dataset in [('train', data_train), ('dev', data_dev)]:
		misclassification_dict = init_misclassification_dict([i for i in range(len(index_to_tag))])
		for p, h, lbl in dataset:
			var_p = m.cuda_wrap(autograd.Variable(p))
			var_h = m.cuda_wrap(autograd.Variable(h))
			var_lbl = m.cuda_wrap(autograd.Variable(lbl))

			prediction = classifier(var_p, var_h)
			_, predicted_idx = torch.max(prediction, dim=1)
			predicted_idx = predicted_idx.data[0]
			lbl = lbl[0]
			misclassification_dict[lbl][predicted_idx] += 1

		# output misclassicication dict
		for lbl_idx_gold in range(len(index_to_tag)):
			for lbl_idx_predicted in range(len(index_to_tag)):
				# add dummy duplicate to have same format as when dimensions are inversed
				print(name + '-' + index_to_tag[lbl_idx_gold] + '-' + index_to_tag[lbl_idx_predicted] + '-' + index_to_tag[lbl_idx_predicted] + ' ' + str(misclassification_dict[lbl_idx_gold][lbl_idx_predicted]))
	# count predictions for gold - predicted pairs


mapper = dict()
mapper['sents'] = print_sents
mapper['plot'] = plot_findings
mapper['print_stats_simple_model'] = print_stats_simple_model
mapper['plot_simple_model'] = plot_stats_at_path

def main():
    args = docopt("""Analyse male/female effects

    Usage: 	male_female_analyse.py sents [<params>]
    		male_female_analyse.py plot [<params>]
    		male_female_analyse.py print_stats_simple_model <params>
    		male_female_analyse.py plot_simple_model <params>
        

    """)

    fn = [k for k in args if args[k] == True][0]
    mapper[fn](args['<params>'])



if __name__ == '__main__':
    main()
