'''Evaluate different adaptions to the sentence representation'''

import train
import config
import model as m
import numpy as np
import torch
import analyse_repr.analyse as analyse
import embeddingholder
import mydataloader
import sys

from docopt import docopt

def run_twists(classifier, data_train, data_dev, padding_token, twister_queue):
	'''
	Runs a series of twisted models, specified in twister_queue. The twisted model
	gets evaluated on train and dev data.

	:param classifier 	 model to use
	:param data_train 	train data
	:param data_dev 	dev data
	:param padding_token 	padding token from embeddings
	:param twister_queue 	[(name-to-print, twister)] to run the twisted models.
	'''
	# Load model, data, embeddings

	for name, twister in twister_queue:
		print(name)
		print('Accuracy (train):', train.evaluate(classifier, data_train, 32, padding_token, twister=twister))
		print('Accuracy (dev):', train.evaluate(classifier, data_dev, 32, padding_token, twister=twister))
		print('')
		sys.stdout.flush()
		# out: train acc, dev acc, train acc delta, dev acc delta


def mirror(v, mirror_point):
	'''
	Mirrors the value on the mirror point, s.t. the distance the point is below mirror_point
	will be the distance the resulting point is above mirror_point and vice versa.

	:param v 		value to mirror
	:param mirror_point 	this is the point to relate to when mirroring.
	'''
	diff = np.absolute(mirror_point - v)
	if v < mirror_point:
		return mirror_point + diff
	else:
		return mirror_point - diff

def flip_dimension(dim, repr, a_set):
	'''
	Flip the dimensional value by mirroring it in the middle between min_value and max_value.
	'''
	mirror_point = a_set.min[dim] + np.absolute(a_set.max[dim] - a_set.min[dim]) / 2
	to_flip = repr[:,dim].data
	flipped = m.cuda_wrap(torch.FloatTensor([mirror(v, mirror_point) for v in to_flip]))
	repr[:,dim] = flipped
	return repr


def eval_mf(classifier, data_train, data_dev, padding_token, a_set):

	def flip_fn(rep, typ, tools):
		a_set = tools[0]
		flip_premise = tools[1]
		flip_hyp = tools[2]

		
		

		# Flip premise representations
		if typ == 'premise':
			for flip_premise_idx in flip_premise:
				rep = flip_dimension(flip_premise_idx, rep, a_set)

		# Flip hypothesis represnetations
		if typ == 'hypothesis':
			for flip_hyp_idx in flip_hyp:
				rep = flip_dimension(flip_hyp_idx, rep, a_set)
		
		return rep


	# invert male dimensions of premise
	t_p_m1 = m.ModelTwister(flip_fn, (a_set, [602], []))
	t_p_m2 = m.ModelTwister(flip_fn, (a_set, [602, 199], []))
	t_p_m3 = m.ModelTwister(flip_fn, (a_set, [602, 199, 280], []))
	t_p_m4 = m.ModelTwister(flip_fn, (a_set, [602, 199, 280, 89], []))

	# invert male dimensions of hypothesis
	t_h_m1 = m.ModelTwister(flip_fn, (a_set, [], [602]))
	t_h_m2 = m.ModelTwister(flip_fn, (a_set, [], [602, 199]))
	t_h_m3 = m.ModelTwister(flip_fn, (a_set, [], [602, 199, 280]))
	t_h_m4 = m.ModelTwister(flip_fn, (a_set, [], [602, 199, 280, 89]))

	# invert male dimensions of premise and hypothesis
	t_ph_m1 = m.ModelTwister(flip_fn, (a_set, [602], [602]))
	t_ph_m2 = m.ModelTwister(flip_fn, (a_set, [602, 199], [602, 199]))
	t_ph_m3 = m.ModelTwister(flip_fn, (a_set, [602, 199, 280], [602, 199, 280]))
	t_ph_m4 = m.ModelTwister(flip_fn, (a_set, [602, 199, 280, 89], [602, 199, 280, 89]))

	# invert female dimensions of premise
	t_p_f1 = m.ModelTwister(flip_fn, (a_set, [1730], []))
	t_p_f2 = m.ModelTwister(flip_fn, (a_set, [1730, 845], []))
	t_p_f3 = m.ModelTwister(flip_fn, (a_set, [1730, 845, 311], []))
	t_p_f4 = m.ModelTwister(flip_fn, (a_set, [1730, 845, 311, 609], []))

	# invert female dimensions of hypothesis
	t_h_f1 = m.ModelTwister(flip_fn, (a_set, [], [1730]))
	t_h_f2 = m.ModelTwister(flip_fn, (a_set, [], [1730, 845]))
	t_h_f3 = m.ModelTwister(flip_fn, (a_set, [], [1730, 845, 311]))
	t_h_f4 = m.ModelTwister(flip_fn, (a_set, [], [1730, 845, 311, 609]))

	# invert female dimensions of premise and hypothesis
	t_ph_f1 = m.ModelTwister(flip_fn, (a_set, [1730], [1730]))
	t_ph_f2 = m.ModelTwister(flip_fn, (a_set, [1730, 845], [1730, 845]))
	t_ph_f3 = m.ModelTwister(flip_fn, (a_set, [1730, 845, 311], [1730, 845, 311]))
	t_ph_f4 = m.ModelTwister(flip_fn, (a_set, [1730, 845, 311, 609], [1730, 845, 311, 609]))

	# invert male&female dimensions of premise
	t_p_mf1 = m.ModelTwister(flip_fn, (a_set, [602, 1730], []))
	t_p_mf2 = m.ModelTwister(flip_fn, (a_set, [602, 199, 1730, 845], []))
	t_p_mf3 = m.ModelTwister(flip_fn, (a_set, [602, 199, 280, 1730, 845, 311], []))
	t_p_mf4 = m.ModelTwister(flip_fn, (a_set, [602, 199, 280, 89, 1730, 845, 311, 609], []))

	# invert male&female dimensions of hypothesis
	t_h_mf1 = m.ModelTwister(flip_fn, (a_set, [], [602, 1730]))
	t_h_mf2 = m.ModelTwister(flip_fn, (a_set, [], [602, 199, 1730, 845]))
	t_h_mf3 = m.ModelTwister(flip_fn, (a_set, [], [602, 199, 280, 1730, 845, 311]))
	t_h_mf4 = m.ModelTwister(flip_fn, (a_set, [], [602, 199, 280, 89, 1730, 845, 311, 609]))

	# invert male&female dimensions of premise and hypothesis
	t_ph_mf1 = m.ModelTwister(flip_fn, (a_set, [602, 1730], [602, 1730]))
	t_ph_mf2 = m.ModelTwister(flip_fn, (a_set, [602, 199, 1730, 845], [602, 199, 1730, 845]))
	t_ph_mf3 = m.ModelTwister(flip_fn, (a_set, [602, 199, 280, 1730, 845, 311], [602, 199, 280, 1730, 845, 311]))
	t_ph_mf4 = m.ModelTwister(flip_fn, (a_set, [602, 199, 280, 89, 1730, 845, 311, 609], [602, 199, 280, 89, 1730, 845, 311, 609]))

	twister_queue = [
	('No twist', None),
	
	('Invert male(1) in premise', t_p_m1), ('Invert male(2) in premise', t_p_m2), ('Invert male(3) in premise', t_p_m3), ('Invert male(4) in premise', t_p_m4),
	('Invert male(1) in hyp', t_h_m1), ('Invert male(2) in hyp', t_h_m2), ('Invert male(3) in hyp', t_h_m3), ('Invert male(4) in hyp', t_h_m4),
	('Invert male(1) in both', t_ph_m1), ('Invert male(2) in both', t_ph_m2), ('Invert male(3) in both', t_ph_m3), ('Invert male(4) in both', t_ph_m4),
	
	('Invert female(1) in premise', t_p_f1), ('Invert female(2) in premise', t_p_f2), ('Invert female(3) in premise', t_p_f3), ('Invert female(4) in premise', t_p_f4),
	('Invert female(1) in hyp', t_h_f1), ('Invert female(2) in hyp', t_h_f2), ('Invert female(3) in hyp', t_h_f3), ('Invert female(4) in hyp', t_h_f4),
	('Invert female(1) in both', t_ph_f1), ('Invert female(2) in both', t_ph_f2), ('Invert female(3) in both', t_ph_f3), ('Invert female(4) in both', t_ph_f4),
	
	('Invert mf(1) in premise', t_p_mf1), ('Invert mf(2) in premise', t_p_mf2), ('Invert mf(3) in premise', t_p_mf3), ('Invert mf(4) in premise', t_p_mf4),
	('Invert mf(1) in hyp', t_h_mf1), ('Invert mf(2) in hyp', t_h_mf2), ('Invert mf(3) in hyp', t_h_mf3), ('Invert mf(4) in hyp', t_h_mf4),
	('Invert mf(1) in both', t_ph_mf1), ('Invert mf(2) in both', t_ph_mf2), ('Invert mf(3) in both', t_ph_mf3), ('Invert mf(4) in both', t_ph_mf4)
	]

	t2_p_m1 = m.ModelTwister(flip_fn, (a_set, [89], []))
	t2_p_m2 = m.ModelTwister(flip_fn, (a_set, [89, 280], []))
	t2_p_m3 = m.ModelTwister(flip_fn, (a_set, [89, 280, 199], []))
	t2_p_m4 = m.ModelTwister(flip_fn, (a_set, [89, 280, 199, 602], []))
	twister_queue2 = [
		('Invert male(1) in premise', t2_p_m1), ('Invert male(2) in premise', t2_p_m2), ('Invert male(3) in premise', t2_p_m3), ('Invert male(4) in premise', t2_p_m4),

	]

	run_twists(classifier, data_train, data_dev, padding_token, twister_queue2)
	#evaluate.evaluate(model_path, data_path, embeddings_path, twister = twister)



def eval_outlier(classifier, data_train, data_dev, padding_token, a_set):
	vals = [(i, a_set.mean[i], a_set.sd[i], a_set.min[i], a_set.max[i], 
		max([np.absolute(a_set.mean[i] - a_set.max[i]), np.absolute(a_set.mean[i] - a_set.max[i])])) for i in range(len(a_set.max))]
	vals = [(i, mean, sd, vmin, vmax, meandiff, meandiff / sd) for i, mean, sd, vmin, vmax, meandiff in vals]

	def crop_val(v, vmin, vmax):
		if v > vmax:
			return vmax
		elif v < vmin:
			return vmin
		else:
			return v

	def crop_fn(rep, typ, tools):
		means = tools[0]
		sd = tools[1]
		factor = tools[2]

		batch_size = rep.size()[0]
		dimens = rep.size()[1]
		for dim in range(dimens):
			vmin = means[dim] - factor * sd[dim]
			vmax = means[dim] + factor * sd[dim]
			cropped_vals = m.cuda_wrap(torch.FloatTensor([crop_val(v, vmin, vmax) for v in rep[:, dim].data]))
			rep[:, dim] = cropped_vals

		return rep

	
	twister_queue = [
		('outlier=15SD', m.ModelTwister(crop_fn, (a_set.mean, a_set.sd, 15))),
		('outlier=12SD', m.ModelTwister(crop_fn, (a_set.mean, a_set.sd, 12))),
		('outlier=10SD', m.ModelTwister(crop_fn, (a_set.mean, a_set.sd, 10))),
		('outlier=9SD', m.ModelTwister(crop_fn, (a_set.mean, a_set.sd, 9))),
		('outlier=8SD', m.ModelTwister(crop_fn, (a_set.mean, a_set.sd, 8))),
		('outlier=6SD', m.ModelTwister(crop_fn, (a_set.mean, a_set.sd, 6))),
		('outlier=5SD', m.ModelTwister(crop_fn, (a_set.mean, a_set.sd, 5)))
	]

	run_twists(classifier, data_train, data_dev, padding_token, twister_queue)







mapper = dict()
mapper['mf'] = eval_mf
mapper['ol'] = eval_outlier


def main():
	args = docopt("""Evaluate on given dataset in terms of accuracy.

	Usage:
		eval_twist.py <model> <data_train> <data_dev> <statpath> <type> [<embeddings>]

		<model> = Path to trained model
		<data>  = Path to data to test model with 
		<embeddings>  = New embedding file to use unknown words from 
	""")

	model_path = args['<model>']
	data_path_train = args['<data_train>']
	data_path_dev = args['<data_dev>']
	embeddings_path = args['<embeddings>']
	eval_type = args['<type>']
	stat_path = args['<statpath>']

	print('# Loading model ...')
	if embeddings_path == None:
		embedding_holder = embeddingholder.EmbeddingHolder(config.PATH_WORD_EMBEDDINGS)
	else:
		embedding_holder = embeddingholder.EmbeddingHolder(embeddings_path)
	classifier, _ = m.load_model(model_path, embedding_holder=embedding_holder)
	m.cuda_wrap(classifier)
	classifier.eval()
	a_set = analyse.AnalyseSet(stat_path)

	print('# Loading data train ...')
	data_train = mydataloader.get_dataset_chunks(data_path_train, embedding_holder, chunk_size=32*400, mark_as='')

	print('# Loading data dev ...')
	data_dev = mydataloader.get_dataset_chunks(data_path_dev, embedding_holder, chunk_size=32*400, mark_as='')

	print('# Evaluate ...')
	mapper[eval_type](classifier, data_train, data_dev, embedding_holder.padding(), a_set)



if __name__ == '__main__':
	main()

    
