import sys
import matplotlib.pyplot as plt
import numpy as np
import random

from docopt import docopt

def print_sd_rank(params):
	folder = params['<folder>']
	pre = folder + 'invert_4m4f_train'
	file_names = [
		pre + '-correct_correct.txt',
		pre + '-correct_incorrect.txt',
		pre + '-incorrect_correct.txt',
		pre + '-incorrect_incorrect.txt'
	]

	def chunker(seq, size):
		return (seq[pos:pos + size] for pos in range(0, len(seq), size))

	all_data_raw = []
	dt = np.dtype(float)
	for file in file_names:
		with open(file) as f_in:
			data = [(d[0], d[1], d[2], d[3], d[4], d[5], d[6].strip().split(' ')) 
				for d in chunker(f_in.readlines(), 9)]
		all_data_raw += data

	# remember all representations
	unique_repr = []
	used_sents = set()
	while len(all_data_raw) > 0:
		current = all_data_raw.pop()
		if current[0] not in used_sents:
			used_sents.add(current[0])
			unique_repr.append(np.asarray(current[2].strip().split(' '), dtype=float))
		if current[3] not in used_sents:
			used_sents.add(current[3])
			unique_repr.append(np.asarray(current[5].strip().split(' '), dtype=float))
	repr_matrix = np.asmatrix(unique_repr)
	sd = np.asarray(np.std(repr_matrix, axis=0)).flatten()
	sd = sorted([(dim, s) for dim, s in enumerate(sd)], key=lambda x: -x[-1])
	print([dim for dim, _  in sd])

def generate_data(params):
	folder = params['<folder>']
	pre = folder + 'invert_4m4f_train'
	file_names = [
		pre + '-correct_correct.txt',
		pre + '-correct_incorrect.txt',
		pre + '-incorrect_correct.txt',
		pre + '-incorrect_incorrect.txt'
	]

	def chunker(seq, size):
		return (seq[pos:pos + size] for pos in range(0, len(seq), size))

	all_data_raw = []
	dt = np.dtype(float)
	for file in file_names:
		with open(file) as f_in:
			data = [(d[0], d[1], d[2], d[3], d[4], d[5], d[6].strip().split(' ')) 
				for d in chunker(f_in.readlines(), 9)]
		all_data_raw += data

	all_data_raw = [(p, p_act, p_rep, h, h_act, h_rep, int(lbl[0]), int(lbl[1])) 
		for p, p_act, p_rep, h, h_act, h_rep, lbl in all_data_raw]


	# Sort by premise
	all_data_raw = sorted(all_data_raw, key=lambda x: x[0])

	# divide samples into categories
	categories = dict()
	categories['correct'] = []
	categories['only_entailment_incorrect'] = []
	categories['only_contradiction_incorrect'] = []
	categories['only_neutral_incorrect'] = []

	incorrect_gold_lbl_to_cat = ['only_neutral_incorrect', 'only_contradiction_incorrect', 'only_entailment_incorrect']


	def find_category(samples):
		# must have all three labels
		if len(list(set([s[-2] for s in samples]))) != 3:
			return None

		incorrect_lbls = [s[-2] for s in samples if s[-2] != s[-1]]
		if len(incorrect_lbls) == 0:
			return 'correct'
		elif len(incorrect_lbls) == 1:
			return incorrect_gold_lbl_to_cat[incorrect_lbls[0]]
		else:
			return None

	while len(all_data_raw) > 0:
		# find subset having same premise
		last_idx = 1
		p  = all_data_raw[0][0]
		broke_out = False
		for i in range(1, len(all_data_raw)):
			if all_data_raw[i][0] != p:
				last_idx = i
				broke_out = True
				break

		if broke_out == False:
			last_idx = len(all_data_raw)

		sub_data = 	all_data_raw[:last_idx]
		del all_data_raw[:last_idx]
		
		# only use if exactly three samples
		if len(sub_data) == 3:
			cat = find_category(sub_data)
			if cat != None:
				# remember samples
				categories[cat].append(sub_data)

	# Select for usage
	def sample(data, amount):
		return [data[i] for i in sorted(random.sample(range(len(data)), amount))]

	sample_correct = sample(categories['correct'], 150)
	sample_only_entailment_incorrect = sample(categories['only_entailment_incorrect'], 50)
	sample_only_contradiction_incorrect = sample(categories['only_contradiction_incorrect'], 50)
	sample_only_neutral_incorrect = sample(categories['only_neutral_incorrect'], 50)

	# write to file
	name_out = folder + 'representation_samples_450_150_150_150.txt'
	idx_to_lbl = ['neutral', 'contradiction', 'entailment']
	with open(name_out, 'w') as f_out:
		# go via all categories
		for sample_set in [sample_correct, sample_only_entailment_incorrect, sample_only_contradiction_incorrect, sample_only_neutral_incorrect]:
			
			# via all groups sharing a premise
			for sample_group in sample_set:

				# via all samples within that group
				for p, p_act, p_rep, h, h_act, h_rep, lbl, _ in sample_group:
					f_out.write(p)
					f_out.write(p_act)
					f_out.write(p_rep)
					f_out.write(h)
					f_out.write(h_act)
					f_out.write(h_rep)
					f_out.write(idx_to_lbl[lbl] + '\n')

def load_sents():
	path = './../analyse_repr_data/representation_samples_450_150_150_150.txt'
	with open(path) as f_in:
		lines = f_in.readlines()

	def chunker(seq, size):
		return (seq[pos:pos + size] for pos in range(0, len(seq), size))

	return [(
			chunk[0].strip().split(' '),
			np.asarray(chunk[1].strip().split(' '), dtype=int),
			np.asarray(chunk[2].strip().split(' '), dtype=float),
			chunk[3].strip().split(' '),
			np.asarray(chunk[4].strip().split(' '), dtype=int),
			np.asarray(chunk[5].strip().split(' '), dtype=float),
			chunk[6].strip()
		) for chunk in chunker(lines, 7)]

def plt_confusion_matrix(matrix, p, h, title):
	fig, ax = plt.subplots()
	cax = ax.imshow(matrix, origin='upper')
	fig.colorbar(cax)
	plt.xticks(np.arange(len(h)), h, rotation=45)
	plt.yticks(np.arange(len(p)), p)
	plt.xlabel('hypothesis')
	plt.ylabel('premise')
	plt.title(title, y=-0.2)
	ax.xaxis.tick_top()
	ax.set_xlabel('hypothesis')    
	ax.xaxis.set_label_position('top') 
	width, height = matrix.shape
	for x in range(width):
		for y in range(height):
			plt.annotate(str(matrix[x,y]), size=6, xy=(y, x), horizontalalignment='center', verticalalignment='center')
	plt.show()


def create_conf_matrix(p, p_act, p_rep, h, h_act, h_rep, score_fn):
	matrix = np.zeros((len(p), len(h)))
	for idx_p in range(len(p)):
		for idx_h in range(len(h)):
			matrix[idx_p, idx_h] = score_fn(idx_p, idx_h, p, p_act, p_rep, h, h_act, h_rep)
	return matrix


def analyse_sent_alignment(params):
	data = load_sents()
	sent_idx = int(params['<sent_idx>'])
	conf_type = params['<conf_type>']

	def score_num_act(idx_p, idx_h, p, p_act, p_rep, h, h_act, h_rep):
		'''Score each index by the amount of activation they share.'''
		return len([i for i in range(len(p_act)) if p_act[i] == idx_p and h_act[i] == idx_h])

	fn_dict = dict()
	fn_dict['nshared'] = ('Amount of same dimension per word', score_num_act)


	sent = data[sent_idx]
	title, fn = fn_dict[conf_type]
	matrix = create_conf_matrix(sent[0], sent[1], sent[2], sent[3], sent[4], sent[5], fn)
	plt_confusion_matrix(matrix, sent[0], sent[3], title)

mapper = dict()
mapper['generate_data'] = generate_data
mapper['sd'] = print_sd_rank
mapper['cm'] = analyse_sent_alignment

def main():
	args = docopt("""Analyse the alignment between premise and hypothesis.
		conf_type can be:
		nshared - number of shared dimension


	Usage:
		analyse-alignment.py generate_data <folder>
		analyse-alignment.py sd <folder>
		analyse-alignment.py cm <sent_idx> <conf_type>

	""")

	fn = [k for k in args if args[k] == True][0]

	mapper[fn](args)


if __name__ == '__main__':
	main()
