'''
This file is only used for further investigation of the male/female dimensions of the trained model.
It is bound to a fixed set of files, if it should be used with other files it jhas to be adopted.
'''


import sys

from docopt import docopt
import numpy as np
import mydataloader
from mydataloader import index_to_tag



def print_sents(params):

	FOLDER = './analyses/'
	FILE_BASE = 'invert_4m4f_'
	FILE_APX_DATA = ['train-', 'dev-']
	FILE_CATEGORIES = [
		('Initially and inverted correct', 'correct_correct'),
		('Initially correct and inverted incorrect', 'correct_incorrect'),
		('Initially incorrect and inverted correct', 'incorrect_correct'),
		('Initially and inverted incorrect', 'incorrect_incorrect')
	]

	if params == None:
		num_sents=100
	else:
		num_sents = int(params)

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
				for line in f_in:
					# check if premise
					if line_idx % 9 == 0:
						premise = line.strip()

					# check if hypothesis
					if line_idx % 9 == 3:
						hypothesis = line.strip()

					# check if label
					if line_idx % 9 == 6:
						labels = [index_to_tag[int(v)] for v in line.strip().split(' ')]
						print('premise:', premise)
						print('hypothesis:', hypothesis)
						print('labels:', labels)
						print('-----')


					line_idx += 1

			


mapper = dict()
mapper['sents'] = print_sents

def main():
    args = docopt("""Analyse male/female effects

    Usage: male_female_analyse.py sents [<params>]
        

    """)

    fn = [k for k in args if args[k] == True][0]
    mapper[fn](args['<params>'])



if __name__ == '__main__':
    main()
