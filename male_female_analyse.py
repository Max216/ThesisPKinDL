'''
This file is only used for further investigation of the male/female dimensions of the trained model.
It is bound to a fixed set of files, if it should be used with other files it jhas to be adopted.
'''


import sys

from docopt import docopt
import numpy as np
import mydataloader
from mydataloader import index_to_tag

import matplotlib.pyplot as plt


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

def init_misclassification_dict(labels):
	d = dict()
	for label in labels:
		d[label] = dict()
		for label2 in labels:
			d[label][label2] = 0
	return d


def plot_correct_incorrect_bar(x_labels, misclassification_dict):
	num_groups = len(x_labels)
	correct = [misclassification_dict[lbl][lbl] for lbl in x_labels]

def plot_findings(params):

	if params == None:
		params = 'train'

	with open(FILE_OUT_TWIST_COUNTS) as f_in:
		all_data = [(line.strip().split(' ')[0], int(line.strip().split(' ')[1])) for line in f_in.readlines()]

	filtered_data = [(key.split('-')[1:], amount) for key, amount in all_data if key.split('-')[0] == params]
	filtered_data = [(k[0], k[1], k[2], amount) for k, amount in filtered_data]
	
	x_labels = index_to_tag
	# Plot basic results for normal model
	misclassification_dict = init_misclassification_dict(x_labels)
	for gold, predicted, _, amount in filtered_data:
		misclassification_dict[gold][predicted] = amount
	plot_correct_incorrect_bar(x_labels, misclassification_dict)

	# plot basic results for inversed model
	misclassification_dict = init_misclassification_dict(x_labels)
	for gold, _, predicted_inv, amount in filtered_data:
		misclassification_dict[gold][predicted_inv] = amount


	# Plot inversed model compared with nomal model
	


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
			cnt_printed = 0
			with open(file) as f_in:
				line_idx = 0
				premise = None
				hypothesis = None
				labels = None
				confidences_normal = None

				for line in f_in:
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

					if line_idx % 9 == 7:
						confidences_inv = line.strip().split(' ')
						lbl_gold = index_to_tag[labels[0]]
						lbl_predicted = index_to_tag[labels[1]]
						lbl_predicted_inv = index_to_tag[labels[2]]
						conf_norml = confidences_normal[labels[1]]
						conf_inv = confidences_inv[labels[2]]

						if cnt_printed < num_sents:
							print('premise:', premise)
							print('hypothesis:', hypothesis)
							print('label-gold:', lbl_gold)
							print('label-predicted:', lbl_predicted, '(' + conf_norml + ')')
							print('label-predicted-inv:', lbl_predicted_inv + '(' + conf_inv + ')')
							print('-----')
							cnt_printed += 1

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

			


mapper = dict()
mapper['sents'] = print_sents
mapper['plot'] = plot_findings

def main():
    args = docopt("""Analyse male/female effects

    Usage: 	male_female_analyse.py sents [<params>]
    		male_female_analyse.py plot
        

    """)

    fn = [k for k in args if args[k] == True][0]
    mapper[fn](args['<params>'])



if __name__ == '__main__':
    main()
