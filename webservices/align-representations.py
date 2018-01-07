from bottle import request, route, run, template, static_file

# stupid thing to import from parent directory
#import os,sys,inspect
#currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
#parentdir = os.path.dirname(currentdir)
#sys.path.insert(0,parentdir) 

import fixed_model_code as m

import os
import nltk
from nltk import word_tokenize

import analyse_alignment as aa

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_options = [
	(
		'Model (small) with fine-tuned padding', 
		'./models/0_0002lr-1600hidden-256_512_1024lstm-32batch-43_1-relu-0_1dropout_2017-11-26_22:13_glove.model',
		'This model is trained on SNLI with sentence encoding BiLSTMs with dimensions: 256 + 512 + 1024. Padding however was fine tuned. All evaluation on the dimensions and representation I did so far use this model.',
		87.42, 85.20, 84.78,
		'normal'
	),
	(
		'Model (small) without fine-tuned padding',
		'./models/0_0002lr-1600hidden-256_512_1024lstm-32batch-43_1-relu-0_1dropout_2017-12-11_11:49_opts:all_zero_padding.tmpsave.model',
		'This model is trained on SNLI with sentence encoding BiLSTMs with dimensions 256 + 512 + 1024. Fixed the bug with fine tuning padding.',
		89.38, 86.01, 85.37,
		'normal'
	),
	(
		'Model (big) without fine-tuned padding',
		'./models/0_0002lr-1600hidden-512_1024_2048lstm-32batch-43_1-relu-0_1dropout_2018-01-01_01:08_opts:zero_padding_big_model_all.model',
		'This model is trained on SNLI with sentence encoding BiLSTMs with dimensions 512 + 1024 + 2048. Needs still more training.',
		89.70, 85.92, '-',
		'normal'
	),
	(
		'Model (small) with ReLu representations',
		'./models/0_0002lr-1600hidden-256_512_1024lstm-32batch-43_1-relu-0_1dropout_2017-12-23_11:48_opts:relu_sent_zero_padding_all.model',
		'This model is trained on SNLI with sentence encoding BiLSTMs with dimensions 256 + 512 + 1024. The resulting representation from max-pooling is sent through ReLu to avoid negative values. Needs still more training.',
		89.44, 85.62, '-',
		'relu'
	),
	(
		'Model (small) with random embeddings',
		'./models/0_0002lr-1600hidden-256_512_1024lstm-32batch-43_1-relu-0_1dropout_2017-12-26_19:54_opts:zero_padding_random_embeddings_all.model',
		'This model is trained on SNLI with sentence encoding BiLSTMs with dimensions 256 + 512 + 1024. Instead of pretrained GloVe embeddings, random embeddings were used (Glorot initialized).',
		85.53, 79.54, '-',
		'rnd_embeddings'
	)
]

general_type_options = [
	('150 samples per label', ['./stored_data/representation_samples_450_150_150_150.txt']),
	('Swapped premise-hypothesis in label entailment', ['./stored_data/swapped_results_entailment_entailment.txt','./../analyses/swapped_results_entailment_contradiction.txt','./../analyses/swapped_results_entailment_neutral.txt']),
	('Swapped premise-hypothesis in label neutral', ['./stored_data/swapped_results_neutral_entailment.txt','./../analyses/swapped_results_neutral_contradiction.txt','./../analyses/swapped_results_neutral_neutral.txt']),
	('Swapped premise-hypothesis in label contradiction', ['./stored_data/swapped_results_contradiction_entailment.txt','./../analyses/swapped_results_contradiction_contradiction.txt','./../analyses/swapped_results_contradiction_neutral.txt'])
]

@route('/predict_representations', method='POST')
def predict_representations():
	model_idx = int(request.json['model'])
	p_rep = request.json['repPremise']
	h_rep = request.json['repHypothesis']
	model_path = model_options[model_idx][1]
	model_type = model_options[model_idx][-1]

	result = aa.predict_rep(model_path, model_type, p_rep, h_rep)
	return result

@route('/alignment_general_results', method='POST')
def evaluate_general():
	bin_size = float(request.forms.get('stepsize_grid'))
	plot_type = int(request.forms.get('general_sample_selection'))
	zero_threshold = None

	if request.forms.get('cb_uncolor_center'):
		zero_threshold = float(request.forms.get('zero-threshold'))
	names = aa.plot_general_statistics(bin_size, zero_threshold, general_type_options[plot_type][1])
	return ';'.join(names)

@route('/alignment_general_sample', method='GET')
def evaluate_general_sample():
	premise = request.query.get('premise')
	hypothesis = request.query.get('hypothesis')
	model_idx = int(request.query.get('model'))
	bin_size = float(request.query.get('bin_size'))
	zero_threshold = None
	if request.query.get('cb_uncolor_center'):
		zero_threshold = float(request.query.get('zero-threshold'))

	model_path = model_options[model_idx][1]
	model_type = model_options[model_idx][-1]
	lbl, activations, representations = aa.test(model_path, model_type, premise, hypothesis)

	sample = aa.Sample(
		word_tokenize(premise), activations[0].data[0], representations[0].data[0], 
		word_tokenize(hypothesis), activations[1].data[0], representations[1].data[0], 
		lbl, lbl)
	return aa.create_json_matrix(sample, bin_size, zero_threshold, dict([('prediction', lbl)]))
	#return '''{"props": {"adapted": [[6, 5, 35], [7, 5, 364], [6, 6, 52], [7, 6, 1375]]}, "plot": {"z": [[0, 1, 0, 0, 0, 2, 1, 1, 0, 0, 1, 0, 2], [0, 1, 0, 0, 0, 9, 1, 1, 1, 0, 1, 1, 0], [0, 0, 0, 0, 1, 1, 2, 2, 0, 1, 0, 2, 0], [1, 0, 0, 1, 0, 5, 0, 2, 1, 0, 0, 0, 0], [0, 0, 0, 2, 1, 6, 4, 3, 0, 1, 1, 0, 0], [0, 0, 1, 0, 2, 11, 5, 0, 1, 1, 1, 0, 1], [0, 0, 0, 0, 3, 30, 30, 9, 2, 0, 2, 0, 3], [0, 0, 1, 2, 9, 30, 30, 29, 10, 3, 3, 4, 1], [0, 0, 0, 0, 9, 30, 5, 1, 0, 0, 0, 0, 0], [0, 1, 0, 2, 3, 3, 0, 1, 0, 0, 0, 0, 0]], "x": ["v: -0.6083", "v: -0.5083", "v: -0.4083", "v: -0.3083", "v: -0.2083", "v: -0.1083", "v: -0.0083", "v: 0.0917", "v: 0.1917", "v: 0.2917", "v: 0.3917", "v: 0.4917", "v: 0.5917"], "y": ["v: 0.6508", "v: 0.5508", "v: 0.4508", "v: 0.3508", "v: 0.2508", "v: 0.1508", "v: 0.0508", "v: -0.0492", "v: -0.1492", "v: -0.2492"]}}'''

@route('/alignment_results', method='POST')
def evaluate():
	model_idx = int(request.forms.get('selected_model_index'))
	premise = request.forms.get('premise')
	hypothesis = request.forms.get('hypothesis')
	feature = request.forms.get('feature_selection')
	shared_or_unshared = request.forms.get('shared_or_not_shared')

	# filters
	filter_single_threshold_enabled = request.forms.get('applied_filters_st') != None
	filter_double_threshold_enabled = request.forms.get('applied_filters_dt') != None

	single_threshold_value = None
	double_threshold_value = None
	if filter_single_threshold_enabled:
		single_threshold_value = float(request.forms.get('single_threshold_value'))
	if filter_double_threshold_enabled:
		double_threshold_value = float(request.forms.get('double_threshold_value'))

	unshared_meaningful = None
	unshared_meaningful_amount = None
	unshared_min_val = None
	if shared_or_unshared == 'not_shared':
		unshared_meaningful = float(request.forms.get('unshared_t_meaningful_dims'))
		unshared_meaningful_amount = int(request.forms.get('unshared_amount_meaningful_dims'))
		unshared_min_val = float(request.forms.get('unshared_single_t'))

	model_path = model_options[model_idx][1]
	model_type = model_options[model_idx][-1]

	lbl, activations, representations = aa.test(model_path, model_type, premise, hypothesis)

	# plotting

	# create sample with dummy labels
	sample = aa.Sample(
		word_tokenize(premise), activations[0].data[0], representations[0].data[0], 
		word_tokenize(hypothesis), activations[1].data[0], representations[1].data[0], 
		lbl, lbl)

	# Filter functions
	def filter_threshold_single(sample):
		filtered_dims = [dim for dim in sample.dims if sample.p_rep[dim] >= single_threshold_value or sample.h_rep[dim] > single_threshold_value]
		return ('t', filtered_dims)

	def filter_threshold_both(sample):
		filtered_dims = [dim for dim in sample.dims if sample.p_rep[dim] >= double_threshold_value and sample.h_rep[dim] > double_threshold_value]
		return ('tb', filtered_dims)

	if filter_single_threshold_enabled:
		sample.filter(filter_threshold_single)
	if filter_double_threshold_enabled:
		sample.filter(filter_threshold_both)

	path = aa.analyse_sent_alignment(sample, feature, unshared_meaningful, unshared_meaningful_amount, unshared_min_val, save='./data/')

	return lbl + ';' + path


@route('/alignments')
def alignments():

	
	params  = {
	'model_options': model_options,
	'selected_model_idx': 1,
	'general_types': general_type_options
	}

	return template('align-representations', params)

@route('/static/<path>')
def server_static(path):
	return static_file(path, root='./webcontent')

@route('/images/<path>')
def image_serve(path):
	return static_file(path, root='./data/') 

run(host='', port=9876, debug=True, reloader=True)