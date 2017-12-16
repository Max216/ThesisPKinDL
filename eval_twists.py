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
import os

from torch.utils.data import DataLoader
import torch.autograd as autograd

from train import CollocateBatchWithSents

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

def find_mf_misclassified_sents(classifier, data_train, data_dev, padding_token, a_set):
	'''
	Find all samples that have normally been classified correctly and now are classified wrong,
	when inverting male/female dimensions.
	'''

	IDX_GOLD = 6
	IDX_PREDICTED = 7
	IDX_PREDICTED_INV = 8

	def flip_fn(rep, typ, tools):
		a_set = tools[0]
		flip_indizes = tools[1]

		if typ == 'hypothesis':
			for idx in flip_indizes:
				rep = flip_dimension(idx, rep, a_set)
		
		return rep

	# do this for train/dev:
	experiment_name = './analyses/invert_4m4f_'
	twist = m.ModelTwister(flip_fn, (a_set, [602, 199, 280, 89, 1730, 845, 311, 609]))
	#twist = m.ModelTwister(flip_fn, (a_set, [1, 2, 3, 4, 5, 6, 7, 8]))
	for name, data_set in [('train', data_train), ('dev', data_dev)]:

		print('# Load data:', name)
		loader = [
			DataLoader(chunk, drop_last = False, batch_size=1, shuffle=False, collate_fn=CollocateBatchWithSents(padding_token)) 
			for chunk in data_set
		]

		correct_before_correct_after = []
		correct_before_incorrect_after = []
		incorrect_before_correct_after = []
		incorrect_before_incorrect_after = []

		for chunk in loader:
			print('classify new chunk')
			# classify normally
			for batch_p, batch_h, batch_lbl, batch_sent_p, batch_sent_h in chunk:
				label_scores_normal, act_indizes, reprs = classifier(
					autograd.Variable(m.cuda_wrap(batch_p), requires_grad=False),
					autograd.Variable(m.cuda_wrap(batch_h), requires_grad=False),
					output_sent_info = True)
				
				_, predicted_indizes_normal = torch.max(label_scores_normal, dim=1)
				label_scores_normal = label_scores_normal.data
				activation_p = act_indizes[0].data
				activation_h = act_indizes[1].data
				reprs_p = reprs[0].data
				reprs_h = reprs[1].data

				# classify with m/f dimensions inverted
				label_scores_inverted = classifier(
					autograd.Variable(m.cuda_wrap(batch_p), requires_grad=False),
					autograd.Variable(m.cuda_wrap(batch_h), requires_grad=False),
					twister = twist).data

				_, predicted_indizes_inverted = torch.max(label_scores_inverted, dim=1)


				# check for gold - classified_normally - classified_inverted
				amount = label_scores_normal.size()[0]
				# (premise repr, premise act, premise words, hyp repr, hyp act, hyp words, lbl gold, lbl predicted normal, lbl predicted inversed, confidence normal, confidence inverse)
				processed_data = [(
					reprs_p[i].cpu(), activation_p[i].cpu(), batch_sent_p[i],
					reprs_h[i].cpu(), activation_h[i].cpu(), batch_sent_h[i],
					batch_lbl.cpu()[i], predicted_indizes_normal[i].data.cpu()[0], predicted_indizes_inverted.cpu()[i],
					label_scores_normal.cpu()[i], label_scores_inverted.cpu()[i]
				) for i in range(amount)]	


				# sort 
				correct_before_correct_after += [d for d in processed_data if d[IDX_GOLD] == d[IDX_PREDICTED] and d[IDX_GOLD] == d[IDX_PREDICTED_INV]]
				correct_before_incorrect_after += [d for d in processed_data if d[IDX_GOLD] == d[IDX_PREDICTED] and d[IDX_GOLD] != d[IDX_PREDICTED_INV]]
				incorrect_before_correct_after += [d for d in processed_data if d[IDX_GOLD] != d[IDX_PREDICTED] and d[IDX_GOLD] == d[IDX_PREDICTED_INV]]
				incorrect_before_incorrect_after += [d for d in processed_data if d[IDX_GOLD] != d[IDX_PREDICTED] and d[IDX_GOLD] != d[IDX_PREDICTED_INV]]

		# write to file
		print('Write to file.')
		file_name = experiment_name + name
		out_arr = [
			('correct_correct', correct_before_correct_after),
			('correct_incorrect', correct_before_incorrect_after),
			('incorrect_correct', incorrect_before_correct_after),
			('incorrect_incorrect', incorrect_before_incorrect_after)
		]

		def stringify_arr(arr):
			return ' '.join([str(v) for v in arr]) + '\n'

		for appendix, samples in out_arr:
			with open(file_name + '-' + appendix + '.txt', 'w') as f_out:
				for p_rep, p_act, p_words, h_rep, h_act, h_words, gold, predicted, predicted_inv, scores, scores_inv in samples:
					f_out.write(stringify_arr(p_words))
					f_out.write(stringify_arr(p_act))
					f_out.write(stringify_arr(p_rep))
					f_out.write(stringify_arr(h_words))
					f_out.write(stringify_arr(h_act))
					f_out.write(stringify_arr(h_rep))
					f_out.write(stringify_arr([gold, predicted, predicted_inv]))
					f_out.write(stringify_arr(scores))
					f_out.write(stringify_arr(scores_inv))

def create_misclassification_dict(labels):
	md = dict()
	for lbl_gold in labels:
		md[lbl_gold] = dict()
		for lbl_predicted_normal in labels:
			md[lbl_gold][lbl_predicted_normal] = dict()
			for lbl_predicted_twisted in labels:
				md[lbl_gold][lbl_predicted_normal][lbl_predicted_twisted] = []
	return md

def eval_output_twist(classifier, experiment_name, twister, dataset, padding_token, labels=[0,1,2]):
	'''
	This will evaluate and output the results of a twist done to the models sentence representation.
	All results will be stored in s specific folder, containing the counting results and samples for 
	each category <gold-label> - <predicted-label-normal> <predicted-label-twisted>

	:param experiment_name 		will result in the folder name where everything will be stored.
	:param twister 				how to change the sentence representation
	:param dataset 				The data that gets evaluated
	'''

	print('# Load data')
	sys.stdout.flush()
	loader = [
		DataLoader(chunk, drop_last = False, batch_size=1, shuffle=False, collate_fn=CollocateBatchWithSents(padding_token)) 
		for chunk in dataset
	]

	print('# Start training')
	sys.stdout.flush()
	classification_dict = create_misclassification_dict(labels)
	for chunk in loader:
		print('# Train new chunk')
		sys.stdout.flush()
		for batch_p, batch_h, batch_lbl, batch_sent_p, batch_sent_h in chunk:
				# Get label scores
				label_scores_normal, act_indizes, reprs = classifier(
					autograd.Variable(m.cuda_wrap(batch_p), requires_grad=False),
					autograd.Variable(m.cuda_wrap(batch_h), requires_grad=False),
					output_sent_info = True)
				_, predicted_indizes_normal = torch.max(label_scores_normal, dim=1)
				label_scores_normal = label_scores_normal.data

				label_scores_twisted = classifier(
						autograd.Variable(m.cuda_wrap(batch_p), requires_grad=False),
						autograd.Variable(m.cuda_wrap(batch_h), requires_grad=False),
						twister = twister).data
				_, predicted_indizes_twisted = torch.max(label_scores_twisted, dim=1)

				gold_label = batch_lbl.cpu()[0]
				predicted_normal_label = predicted_indizes_normal.data.cpu()[0]
				predicted_twisted_label = predicted_indizes_twisted.cpu()[0]

				sample = (' '.join(batch_sent_p[0]), ' '.join(batch_sent_h[0]))
				classification_dict[gold_label][predicted_normal_label][predicted_twisted_label].append(sample)

	# write out samples
	directory = './analyses/' + experiment_name
	if not os.path.exists(directory):
		os.makedirs(directory)

	count_results = []
	for lbl_gold in classification_dict.keys():
		for lbl_pred in classification_dict[lbl_gold].keys():
			for lbl_pred_twist in classification_dict[lbl_gold][lbl_pred].keys():
				name_gold = mydataloader.index_to_tag[lbl_gold]
				name_pred = mydataloader.index_to_tag[lbl_pred]
				name_pred_twisted = mydataloader.index_to_tag[lbl_pred_twist]
				name = '-'.join([name_gold, name_pred, name_pred_twisted])
				count_results.append((name, len(classification_dict[lbl_gold][lbl_pred][lbl_pred_twist])))

				# write samples
				with open(directory + '/' + name + '.txt', 'w') as f_out:
					for p, h in classification_dict[lbl_gold][lbl_pred][lbl_pred_twist]:
						f_out.write(p + '\n')
						f_out.write(h + '\n')

	# write our stats
	with open(directory + '/results.txt', 'w') as f_out:
		for name, cnt in count_results:
			f_out.write(name + ' ' + str(cnt) + '\n')



def eval_no_negative_values(appendix, classifier, data, padding_token):
	def change_repr(rep, typ, tools):
		mask = tools
		new_rep = autograd.Variable(m.cuda_wrap(torch.max(rep.data, tools)))
		return new_rep

	experiment_name = 'No negative values' + appendix
	mask = m.cuda_wrap(torch.zeros(1, classifier.dimen_sent_repr).float())
	twister = m.ModelTwister(change_repr, mask)
	eval_output_twist(classifier, experiment_name, twister, data, padding_token)





def eval_male_female_to_neutral_plural(appendix, classifier, data, padding_token):

	def change_repr(rep, typ, tools):
		gender_dims = tools[0]
		genderless_plural_dims = tools[1]

		# get min of gender dims
		rep_vals = rep[0]
		min_gender = torch.min((torch.Tensor([rep_vals[dim].data[0] for dim in gender_dims])))

		# change plural dims appropriately
		copied_rep = autograd.Variable(rep.data.clone().copy_(rep.data))
		for plural_dim in genderless_plural_dims:
			new_val = torch.max(m.cuda_wrap(torch.Tensor((rep_vals[plural_dim].data[0], min_gender))))
			copied_rep[:,plural_dim] = new_val
		
		return copied_rep 

	experiment_name = 'match mf to genderless combined' + appendix
	twister = m.ModelTwister(change_repr, ([602, 199, 280, 89, 1730, 845, 311, 609], [1449, 821]))
	#twister = m.ModelTwister(change_repr, ([3,4,5,6,7], [1,2]))
	eval_output_twist(classifier, experiment_name, twister, data, padding_token)


mapper = dict()
mapper['mf'] = eval_mf
mapper['ol'] = eval_outlier
mapper['misclassified_sents_mf'] = find_mf_misclassified_sents

output_mapper = dict()
output_mapper['mfn'] = eval_male_female_to_neutral_plural
output_mapper['noneg'] = eval_no_negative_values


def main():
	args = docopt("""Evaluate on given dataset in terms of accuracy.

	Usage:
		eval_twist.py normal <model> <data_train> <data_dev> <statpath> <type> [<embeddings>]
		eval_twist.py output <model> <data> <type> <appendix>

		<model> = Path to trained model
		<data>  = Path to data to test model with 
		<embeddings>  = New embedding file to use unknown words from 
	""")

	model_path = args['<model>']
	data_path_train = args['<data_train>']
	data_path_dev = args['<data_dev>']
	data_path_single = args['<data>']
	embeddings_path = args['<embeddings>']
	eval_type = args['<type>']
	stat_path = args['<statpath>']
	appendix = args['<appendix>']

	print('# Loading model ...')
	if embeddings_path == None:
		embedding_holder = embeddingholder.EmbeddingHolder(config.PATH_WORD_EMBEDDINGS)
	else:
		embedding_holder = embeddingholder.EmbeddingHolder(embeddings_path)
	classifier, _ = m.load_model(model_path, embedding_holder=embedding_holder)
	m.cuda_wrap(classifier)
	classifier.eval()

	if args['output'] != None:
		data = mydataloader.get_dataset_chunks(data_path_single, embedding_holder, chunk_size=32*400, include_sent=True)
		output_mapper[eval_type](appendix, classifier, data, embedding_holder.padding())
	else:
		a_set = analyse.AnalyseSet(stat_path)

		include_sent = False
		if eval_type in ['misclassified_sents_mf']:
			include_sent = True

		print('# Loading data train ...')
		data_train = mydataloader.get_dataset_chunks(data_path_train, embedding_holder, chunk_size=32*400, mark_as='', include_sent=include_sent)

		print('# Loading data dev ...')
		data_dev = mydataloader.get_dataset_chunks(data_path_dev, embedding_holder, chunk_size=32*400, mark_as='', include_sent=include_sent)

		print('# Evaluate ...')
		mapper[eval_type](classifier, data_train, data_dev, embedding_holder.padding(), a_set)



if __name__ == '__main__':
	main()

    
