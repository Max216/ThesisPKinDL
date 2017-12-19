from bottle import request, route, run, template, static_file

# stupid thing to import from parent directory
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import fixed_model_code as m
import evaluate as evaluate_lib

import nltk
from nltk import word_tokenize

import analyse_repr
from analyse_repr import analyse_alignment as aa

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

model_options = [
	(
		'Model (small) with fine-tuned padding', 
		'./../models/0_0002lr-1600hidden-256_512_1024lstm-32batch-43_1-relu-0_1dropout_2017-11-26_22:13_glove.model',
		'This model is trained on SNLI with sentence encoding BiLSTMs with dimensions: 256 + 512 + 1024. Padding however was fine tuned. All evaluation on the dimensions and representation I did so far use this model.',
		87.42, 85.20, 84.78
	),
	(
		'Model (small) without fine-tuned padding',
		'./../models/0_0002lr-1600hidden-256_512_1024lstm-32batch-43_1-relu-0_1dropout_2017-12-11_11:49_opts:all_zero_padding.tmpsave.model',
		'This model is trained on SNLI with sentence encoding BiLSTMs with dimensions 256 + 512 + 1024. Fixed the bug with fine tuning padding.',
		89.38, 86.01, 85.37
	)
]

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

	lbl, activations, representations = evaluate_lib.test(model_path, premise, hypothesis)

	# plotting
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
	'selected_model_idx': 1
	}

	return template('align-representations', params)

@route('/static/<path>')
def server_static(path):
	return static_file(path, root='./')

@route('/images/<path>')
def image_serve(path):
	return static_file(path, root='./data/')

run(host='localhost', port=9876, debug=True)