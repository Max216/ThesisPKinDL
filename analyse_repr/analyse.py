import sys
import matplotlib.pyplot as plt

from docopt import docopt
import numpy as np

import analyse_lib

def main():
    args = docopt("""Analyse a model

    Usage:
        analyse.py <path> general
        analyse.py <path> positional [--details=<dim>] [--save]
        analyse.py <path> positional [--find=<q>] 
        analyse.py <path> simple_pos [--details=<dim>] [--save] [--filter=<filter_q>]
        analyse.py <path> simple_pos [--find=<q>] [--filter=<filter_q>]
        analyse.py <path> simple_pos [--stats] [--filter=<filter_q>] [--save]
        analyse.py <path> verb_pos [--details=<dim>] [--save] [--filter=<filter_q>]
        analyse.py <path> verb_pos [--find=<q>] [--filter=<filter_q>]
        analyse.py <path> verb_pos [--stats] [--filter=<filter_q>] [--save]
        analyse.py <path> nn_jj_pos [--details=<dim>] [--save] [--filter=<filter_q>]
        analyse.py <path> nn_jj_pos [--find=<q>] [--filter=<filter_q>]
        analyse.py <path> nn_jj_pos [--stats] [--filter=<filter_q>] [--save]
        analyse.py <path> mcw [--details=<dim>] [--save] [--filter=<filter_q>]
        analyse.py <path> mcw [--find=<q>] [--filter=<filter_q>]
        analyse.py <path> mcw [--stats] [--filter=<filter_q>] [--save]
        analyse.py <path> words --w=<w> [--details=<dim>] [--save] [--filter=<filter_q>]
        analyse.py <path> words --w=<w> [--find=<q>] [--filter=<filter_q>]
        analyse.py <path> words --w=<w> [--stats] [--filter=<filter_q>] [--save]
        analyse.py <path> words --w=<w> --group [--save]
        analyse.py <path> words --g=<g> [--details=<dim>] [--save] [--filter=<filter_q>]
        analyse.py <path> words --g=<g> [--find=<q>] [--filter=<filter_q>]
        analyse.py <path> words --g=<g> [--stats] [--filter=<filter_q>] [--save]
        analyse.py <path> words --g=<g> --group [--save]
        analyse.py <path> words --g=<g> [--details=<dim>] [--save] [--filter=<filter_q>]
        analyse.py <path> words --g=<g> [--find=<q>] [--filter=<filter_q>]
        analyse.py <path> words --g=<g> [--stats] [--filter=<filter_q>] [--save]
        analyse.py <path> words --g=<g> --group [--save]


        <path> = path to sentences with activation
        --details=<dim> = if set, plot details of dimension
        --find=q>		to find the most occurences of q
        --filter=<filter_q> only look in sentences containing filter_q
        --save  store to a file

    """)

    path = args['<path>']
    print(args)

    fn = [k for k in args if args[k] == True][0]
    a_set = AnalyseSet(path)
    print('Script started.')
    details = args['--details']
    q = args['--find']
    filter_q = args['--filter']
    save = args['--save']
    show_stats = args['--stats']
    w = args['--w']
    g = args['--g']
    group = args['--group']

    params = dict()
    params['w_list'] = w
    params['g_list'] = g
    params['group'] = group
    if details == None:
        analyse_lib.tools[fn](a_set, q=q, save=save, filter_q=filter_q, show_stats=show_stats, params=params)
    else:
        analyse_lib.tools[fn](a_set, dim=int(details), save=save, filter_q=filter_q,params=params)



class AnalyseSet:

	def __init__(self, path):
		
		# meta data
		meta_lines = []
		with open(path) as f_in:
			for line in f_in:
				meta_lines.append(line.strip())
				if line.strip() == '# CONTENT':
					break

			all_lines = [line.strip() for line in f_in.readlines()]

				

		self.parse_meta(meta_lines)
		self.parse_sents(all_lines)


	def parse_meta(self, meta_lines):
		self.model_name, self.data_name = meta_lines[0].split(':', 1)[1].split(';DATA:')
		# SENTS:1000;LEN:8
		self.len, self.sent_len = (int(v) for v in meta_lines[1].split(':', 1)[1].split(';LEN:'))
		# skip empty line
		self.mean = self.float_array(meta_lines[3])
		self.sd = self.float_array(meta_lines[4])
		self.min = self.float_array(meta_lines[5])
		self.max = self.float_array(meta_lines[6])


	def parse_sents(self, data_lines):
		# list[start:stop:step]
		self.sents = [sent.split(' ') for sent in data_lines[0::5]]
		self.pos = [pos_sent.split(' ') for pos_sent in data_lines[1::5]]
		self.parses = data_lines[2::5]
		self.activations = [self.int_array(line) for line in data_lines[3::5]]
		self.representations = [self.float_array(line) for line in data_lines[4::5]]
		
		self.sent_repr_dim = len(self.activations[0])

	def get(self, sidx, repr_indizes=-1):
		'''
		Get all information about a sentence. Only wrt the representation indizes specified.
		'''
		if repr_indizes == -1:
			repr_indizes = [i for i in range(len(self.activations))]

		return (
			self.sents[sidx], 
			self.pos[sidx],
			self.parses[sidx], 
			np.take(self.activations[sidx], repr_indizes), 
			np.take(self.representations[sidx], repr_indizes)
		)	

	def float_array(self, line):
		return np.fromstring(line, dtype=float, sep= ' ')

	def int_array(self, line):
		return np.fromstring(line, dtype=int, sep=' ')

	def dimension_rank(self, dim):
		'''According to sd'''
		indizes = sorted(np.arange(self.sent_repr_dim), key=lambda x: -self.sd[x])

		if isinstance(dim, int):
			return indizes.index(dim) + 1
		else:
			# assume iterable
			return [indizes.index(d) + 1 for d in dim]


	def enum_repr(self, sent_indizes, maximum=-1, sort_by='positional'):
		'''
		@return an array with the dimension indizes.

		@param maximum		maximum limit of dimensions (-1=default means no limit)
		@param sort_by	indizes are sorted by position ("positional"[default]) or standard deviation ("sd")
						or function(act) return <int> for each dimension 		
		'''

		if maximum == -1:
			maximum = self.sent_repr_dim

		def sort_by_sd(x):
			''' Sort descending by standard deviation. '''
			return -self.sd[x]

		indizes = None
		if sort_by == 'positional':
			indizes = [i for i in np.arange(self.sent_repr_dim)]
		elif sort_by == 'sd':
			indizes = sorted(np.arange(self.sent_repr_dim), key=sort_by_sd)
		elif callable(sort_by):
			dim_wise_scores = []
			for dim in range(self.sent_repr_dim):
				# activation per sentence of this dimension
				dim_act = np.squeeze(np.asarray(np.matrix(self.activations)[:,dim]))
				# only use specified sentences
				dim_act = np.take(dim_act, sent_indizes)
				dim_wise_scores.append((dim, sort_by(sent_indizes, dim_act)))
			
			sorted_scores = sorted(dim_wise_scores, key=lambda x: x[1], reverse=True)
			indizes = [dim for dim, score in sorted_scores]
		else:
			raise Exception('sort_by must either be "positional" or "sd" or function.')

		return indizes[:maximum]

	def get_dim(self, sent_indizes, dim):
		relevant_activations = [self.activations[idx] for idx in sent_indizes]
		relevant_representations = [self.representations[idx] for idx in sent_indizes]
		
		dim_act = [a[dim] for a in relevant_activations]
		dim_repr = [r[dim] for r in relevant_representations]

		return (dim_act, dim_repr)

	def get_word_along_dim(self, sent_indizes, dim_activations):
		return [self.sents[sent_idx][dim_activations[a_idx]] for a_idx, sent_idx in enumerate(sent_indizes)]
		#return [self.sents[i][dim_activations[i]] for i in range(len(dim_activations))]

	def get_pos_along_dim(self, sent_indizes, dim_activations):
		return [self.pos[sent_idx][dim_activations[a_idx]] for a_idx, sent_idx in enumerate(sent_indizes)]
		#return [self.pos[i][dim_activations[i]] for i in range(len(dim_activations))]

	def enum_sents(self, filter_fn=None):
		indizes = np.arange(len(self.sents))
		if filter_fn != None:
			indizes = np.array([idx for idx in indizes if filter_fn(idx)])

		return indizes


if __name__ == '__main__':
    main()
