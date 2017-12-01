import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
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
'#fffac8', 
'#800000', 
'#808000', 
'#000080', 
'#808080', 
'#e6beff',
'#aaffc3']

def adjust_plot_size(width=30, height=30):
	fig_size = plt.rcParams["figure.figsize"]
	fig_size[0] = width
	fig_size[1] = height
	plt.rcParams["figure.figsize"] = fig_size

def general(a_set, q=None, save=None, filter_q=None):
	'''Plot mean, variance, min, max of a sentence repr'''
	adjust_plot_size(12,12)
	x = a_set.enum_repr(np.arange(len(a_set.sents)))
	y = a_set.mean
	e = a_set.sd

	plt.errorbar(x, y, e, linestyle='None', marker='.', color='green', ecolor='red', elinewidth=0.1, markersize=0.5, alpha=1)
	plt.xlabel('dimension index')
	plt.ylabel('value of representation')
	name = 'Mean and standard deviation over all examined samples'
	plt.title(name)
	plt.savefig('./plots/' + name +'.png')
	plt.clf()

	# plot bar char
	bins_x = np.arange(0, 0.5, 0.005)
	
	bin_dict = dict()
	for b in bins_x:
		bin_dict[b] = len([v for v in a_set.sd if v >= b and v < b+0.005])

	bins_y = [bin_dict[x] for x in bins_x]
	print('>>', sum([bin_dict[x] for x in bins_x if x >= 0.1]))
	width = 0.001
	plt.bar(bins_x, bins_y, width,  color="blue")
	plt.xlabel('standard deviation')
	plt.ylabel('# dimensions')
	name = 'Dimensions per standard deviation'
	plt.title(name)
	plt.savefig('./plots/' + name +'.png')
	plt.clf()

def plot_grid(a_set, sort_by, name, settings, dim_amount=300, sent_amount=300):
	'''
	Plot the grid over all sentences to see what word troggers what dimension.
	'''
	adjust_plot_size()
	
	# only use those sentences for evaluation
	filter_fn = None
	if settings.filter_on:
		filter_fn = settings.filter_fn

	working_sent_idx = a_set.enum_sents(filter_fn)

	# only display those
	sent_idx = working_sent_idx[:sent_amount]

	repr_idx = a_set.enum_repr(working_sent_idx, maximum=dim_amount, sort_by=sort_by)[:dim_amount][::-1]
	r_idx_to_pos = {r_idx: i for (i, r_idx) in enumerate(repr_idx)}
	#print(repr_idx)

	sent_info = [a_set.get(sidx, repr_idx) for sidx in sent_idx]
	colored_sents = [settings.color_sent(w,pos,parse,a,r,repr_idx) for w,pos,parse,a,r in sent_info]
	colors = settings.colors()
	

	# scatter each color
	fig, ax = plt.subplots()


	# Over all labels w/ colors
	color_keys = sorted(list(colors.keys()))
	for lbl in color_keys:

		x_vals = []
		y_vals = []
		color = colors[lbl]

		# Over each sentence with sentence idx
		for x in range(len(colored_sents)):

			cs = colored_sents[x]

			# Over all different groups within each sentence
			for r_idx in cs[lbl]:
				x_vals.append(x)
				y_vals.append(r_idx_to_pos[r_idx])

		ax.scatter(x_vals, y_vals, c=color, s=4, label=lbl, alpha=1.0, edgecolors='none')

	ax.legend()
	y_sd_rank = a_set.dimension_rank(repr_idx)
	y_ticks = ['-sd-rank:' + str(y_sd_rank[i]) + '  -  ' + str(idx) for i,idx in enumerate(repr_idx)]
	plt.yticks([i for i in range(len(repr_idx))], y_ticks)
	plt.title(name)
	plt.xlabel('sentences')
	plt.ylabel('dimensions')
	for label in ax.get_yticklabels():
		label.set_fontsize(6)
	plt.savefig('./plots/' + name +'.png')
	print('saved:', name)
	plt.clf()


def plot_general_stats(a_set, name, settings, save):

	labels = sorted(list(settings.colors().keys()))
	colors = [settings.colors()[k] for k in labels]
	x_vals = np.arange(len(labels))

	filter_fn = None
	if settings.filter_on:
		filter_fn = settings.filter_fn

	working_sent_idx = a_set.enum_sents(filter_fn)

	plt.figure(200)
	adjust_plot_size(width=10, height=6)
	y_vals_dict = settings.count_fn(working_sent_idx, labels)
	y_vals = [y_vals_dict[lbl] for lbl in labels]

	used_samples = len(working_sent_idx)
	total_samples = len(a_set.sents)
	 
	plt.bar(x_vals, y_vals, align='center', alpha=1.0, color=colors)
	plt.xticks(x_vals, labels)
	plt.ylabel('# sentences (' + str(used_samples) + ' / ' + str(total_samples) + ')')
	plt.xlabel('categories')
	plt.title(name)
	if save:
		plt.savefig('./plots/' + name +'.png')
	plt.show()

	

def plot_dim_details(a_set, settings, dim, title, save):

	labels = sorted(list(settings.colors().keys()))
	colors = [settings.colors()[k] for k in labels]
	x_vals = np.arange(len(labels))
	title = title + ' dim=' + str(dim) + ', sd-rank=' + str(a_set.dimension_rank(dim))

	filter_fn = None
	if settings.filter_on:
		filter_fn = settings.filter_fn

	working_sent_idx = a_set.enum_sents(filter_fn)

	def plot_distribution():
		plt.figure(200)
		adjust_plot_size(width=10, height=6)
		y_vals_dict = settings.distribution(working_sent_idx, dim)
		y_vals = [y_vals_dict[lbl] for lbl in labels]

		used_samples = len(working_sent_idx)
		total_samples = len(a_set.sents)
		 
		plt.bar(x_vals, y_vals, align='center', alpha=1.0, color=colors)
		plt.xticks(x_vals, labels)
		plt.ylabel('# sentences (' + str(used_samples) + ' / ' + str(total_samples) + ')')
		plt.xlabel('categories')
		plt_title =  title + ' (distributional)'
		plt.title(plt_title)
		plt.show(block=False)

		if save:
			plt.savefig('./plots/' + plt_title +'.png')


	def plot_stats():
		plt.figure(300)
		activations, representations = a_set.get_dim(working_sent_idx, dim)
		stats_dict = settings.stats(dim, activations, representations, working_sent_idx)

		# might be less if no activation from one group
		stats_labels = [lbl for lbl in labels if lbl in stats_dict]

		mean = [stats_dict[lbl][0] for lbl in stats_labels] + [a_set.mean[dim]]
		sd = [stats_dict[lbl][1] for lbl in stats_labels] + [a_set.sd[dim]]
		vmin = [stats_dict[lbl][2] for lbl in stats_labels] + [a_set.min[dim]]
		vmax = [stats_dict[lbl][3] for lbl in stats_labels] + [a_set.max[dim]]
		x_vals = np.arange(len(stats_labels) + 1)

		plt_mean = plt.errorbar(x_vals, mean, sd, linestyle='None', marker='.', color='green', ecolor='red', elinewidth=1, markersize=5, alpha=1)
		plt_min = plt.scatter(x_vals, vmin, s=5, alpha=1)
		plt_max = plt.scatter(x_vals, vmax, s=5, alpha=1)

		plt.legend([plt_mean, plt_min, plt_max], ['mean with sd', 'min values', 'max values'], bbox_to_anchor=(0,1.1,1,0.2), loc="lower left",mode="expand", borderaxespad=0, ncol=3)
		plt.subplots_adjust(top=0.8)
		plt.xticks(x_vals, stats_labels + ['[DIM=' + str(dim) + ']'])
		plt.xlabel('categories')
		plt.ylabel('value of representation')
		plt_title = title + ' (statistics)'
		plt.title(plt_title)

		adjust_plot_size(width=9, height=6)
		if save:
			plt.savefig('./plots/' + plt_title +'.png')

		plt.show()


	plot_distribution()
	plot_stats()
	plt.clf()

def words_analysis(a_set, name, word_fn, exclude=None, dim=None, save=False, q=None, filter_q=None, show_stats=False):
	pass

def most_common_words(a_set, dim=None, save=False, q=None, filter_q=None, show_stats=False):
	def simplify(pos):
		if pos.startswith('JJ'):
			return 'JJ'
		elif pos.startswith('NN'):
			return 'NN'
		elif pos.startswith('VB'):
			return 'VB'
		elif pos.startswith('PRP'):
			return 'PRP'
		elif pos.startswith('RB'):
			return 'RB'
		else:
			return pos

	
	name = 'Simplified POS'
	pos(a_set, name, simplify, exclude=None, dim=dim, save=save, q=q, filter_q=filter_q, show_stats=show_stats)

def pos(a_set, name, pos_fn, exclude=None, dim=None, save=False, q=None, filter_q=None, show_stats=False):
	labels = sorted(list(set([pos_fn(p) for ps in a_set.pos for p in ps])))

	def colors():
		''' Map category to color '''
		colors = {lbl : color_palette[i] for i, lbl in enumerate(labels)}
		return colors

	def color_sent(words, pos, parse, act, repr, indizes):
		''' Create dict() for sent with k=category, v=[indizes] '''
		result = dict()
		for lbl in labels:
			result[lbl] = [idx for i, idx in enumerate(indizes) if pos_fn(pos[act[i]]) == lbl]
		return result

	def distribution(sent_indizes, dim):
		''' Create dict() for activations with k=category, v=len(activations) '''
		act, _ = a_set.get_dim(sent_indizes, dim)
		words = a_set.get_word_along_dim(sent_indizes, act)
		pos = [pos_fn(p) for p in a_set.get_pos_along_dim(sent_indizes, act)]
		result = dict()
		for lbl in labels:
			result[lbl] = [(p, words[i]) for i,p in enumerate(pos) if p == lbl]

		# print words
		print_dist(words, result)

		for k in result.keys():
			result[k] = len(result[k])

		return result

	def stats(dim, activations, representations, sent_indizes):
		'''Create dict with k=category, v=(mean, sd, min, max)'''
		result = dict()
		pos = [pos_fn(p) for p in a_set.get_pos_along_dim(sent_indizes, activations)]

		# remember repr values per category
		for lbl in labels:
			result[lbl] = [representations[i] for i in range(len(representations)) if lbl == pos[i]]

		for lbl in labels:
			r = np.array(result[lbl])
			if r.shape[0] > 0:
				mean = np.mean(r)
				sd = np.std(r)
				abs_min = np.amin(r)
				abs_max = np.amax(r)

				result[str(lbl)] = (mean, sd, abs_min, abs_max)
			else:
				# rm key
				result.pop(str(lbl), None)

		return result

	def priority_fn(sent_indizes, activations):
		'''return a score for the dimension for having one dominant category'''
		pos = [pos_fn(p) for p in a_set.get_pos_along_dim(sent_indizes, activations)]	
		if exclude != None:
			pos = [p for p in pos if p != exclude]	

		_, most_freq = Counter(pos).most_common(1)[0]
		return most_freq

	def query_fn(sent_indizes, activations):
		'''return a score for the dimension for containing the query'''
		pos = [pos_fn(p) for p in a_set.get_pos_along_dim(sent_indizes, activations)]		
		return len([p for p in pos if p == q])

	def filter_fn(sent_idx):
		return filter_q in [pos_fn(p) for p in a_set.pos[sent_idx]]

	def count_fn(working_sent_idx, labels):
		'''Count Occurences for each'''
		all_pos = [pos_fn(a_set.pos[sent_idx][pos_idx]) for sent_idx in working_sent_idx for pos_idx in range(len(a_set.pos[sent_idx]))]
		counter = Counter(all_pos)
		return counter


	settings = GridSettings(color_sent, colors, distribution, stats, filter_fn, count_fn)
	if filter_q != None:
		settings.filter(True)
		name +='[filter=' + str(filter_q) + ']'

	if show_stats:
		plot_general_stats(a_set, name + ' Overview', settings, save)
	elif dim == None and q == None:
		plot_grid(a_set, 'positional',name + ' of 300 dimensions (first dimensions)', settings)
		plot_grid(a_set, 'sd', name + ' of 300 dimensions (most SD)', settings)
		plot_grid(a_set, priority_fn, name + ' of 300 dimensions (most activations per single position)', settings)
	elif q != None:	
		plot_grid(a_set, query_fn, name + ' of 300 dimensions (most of:' + q + ')', settings)
	else:
		plot_dim_details(a_set, settings, dim, name, save)

def simple_pos(a_set, dim=None, save=False, q=None, filter_q=None, show_stats=False):

	def simplify(pos):
		if pos.startswith('JJ'):
			return 'JJ'
		elif pos.startswith('NN'):
			return 'NN'
		elif pos.startswith('VB'):
			return 'VB'
		elif pos.startswith('PRP'):
			return 'PRP'
		elif pos.startswith('RB'):
			return 'RB'
		else:
			return pos

	
	name = 'Simplified POS'
	pos(a_set, name, simplify, exclude=None, dim=dim, save=save, q=q, filter_q=filter_q, show_stats=show_stats)

def verb_pos(a_set, dim=None, save=False, q=None, filter_q=None, show_stats=False):

	def only_verbs(pos):
		if pos.startswith('V'):
			return pos
		else:
			return 'OTHER'

	name = 'Verb POS'
	pos(a_set, name, only_verbs, exclude='OTHER', dim=dim, save=save, q=q, filter_q=filter_q, show_stats=show_stats)

def nn_jj_pos(a_set, dim=None, save=False, q=None, filter_q=None, show_stats=False):

	def only_nn_jj(pos):
		if pos.startswith('NN') or pos.startswith('JJ'):
			return pos
		else:
			return 'OTHER'

	name = 'Noun - Adj POS'
	pos(a_set, name, only_nn_jj, exclude='OTHER', dim=dim, save=save, q=q, filter_q=filter_q, show_stats=show_stats)
	


def positional(a_set, dim=None, save=False, q=None, filter_q=None, show_stats=False):
	labels = [i for i in range(a_set.sent_len)]

	def colors():
		''' Map category to color '''
		# 8 positions
		colors = {str(k) : color_palette[i] for i, k in enumerate(labels)}
		return colors

	def color_sent(words, pos, parse, act, repr, indizes):
		''' Create dict() for sent with k=category, v=[indizes] '''
		result = dict()
		for lbl in labels:
			result[str(lbl)] = [idx for i, idx in enumerate(indizes) if act[i] == lbl]
		return result

	def distribution(sent_indizes, dim):
		''' Create dict() for activations with k=category, v=len(activations) '''
		act, _ = a_set.get_dim(sent_indizes, dim)
		words = a_set.get_word_along_dim(sent_indizes, act)
		result = dict()
		for lbl in labels:
			result[str(lbl)] = [(a, words[i]) for i,a in enumerate(act) if a == lbl]

		# print words

		print_dist(words, result)

		for k in result.keys():
			result[k] = len(result[k])

		return result

	def stats(dim, activations, representations, sent_indizes):
		'''Create dict with k=category, v=(mean, sd, min, max)'''
		result = dict()

		# remember repr values per category
		for lbl in labels:
			result[str(lbl)] = [representations[i] for i in range(len(representations)) if lbl == activations[i]]

		for lbl in labels:
			r = np.array(result[str(lbl)])
			if r.shape[0] > 0:
				mean = np.mean(r)
				sd = np.std(r)
				abs_min = np.amin(r)
				abs_max = np.amax(r)

				result[str(lbl)] = (mean, sd, abs_min, abs_max)
			else:
				# rm key
				result.pop(str(lbl), None)

		return result

	def priority_fn(sent_indizes, activations):
		'''return a score for the dimension'''		
		most_freq = np.bincount(activations).argmax() 
		return len([a for a in activations if a == most_freq])

	def query_fn(sent_indizes, act):
		return len([a for a in act if a == int(q)])

	settings = GridSettings(color_sent, colors, distribution, stats)

	if dim == None and q == None:
		plot_grid(a_set, 'positional', 'Word position of 300 dimensions (first dimensions)', settings)
		plot_grid(a_set, 'sd', 'Word position of 300 dimensions (most SD)', settings)
		plot_grid(a_set, priority_fn, 'Word position of 300 dimensions (most activations per single position)', settings)
	elif q != None:
		plot_grid(a_set, query_fn, 'Word position of 300 dimensions (most activation from:' + str(q) + ')', settings)
	else:
		plot_dim_details(a_set, settings, dim, 'Word position', save)


def print_dist(words, act_dict):
	categories  =  sorted(list(act_dict.keys()))
	cnt_dicts = [dict() for i in range(len(categories))]
	for i, cat in enumerate(categories):
		for _, w in act_dict[cat]:
			if w in cnt_dicts[i]:
				cnt_dicts[i][w] += 1
			else:
				cnt_dicts[i][w] = 1

	for i,d in enumerate(cnt_dicts):
		print(categories[i] + '(' + str(len(act_dict[categories[i]])) + ')')
		pairs = sorted([(w, cnt) for w, cnt in d.items()], key=lambda x: x[-1], reverse=True)
		print(pairs)
		print('')


class GridSettings:
	def __init__(self, color_sent, colors, distribution, stats, filter_fn=None, count_fn=None):
		self.color_sent = color_sent
		self.colors = colors
		self.distribution = distribution
		self.stats = stats
		self.filter_fn = filter_fn
		self.filter_on = False
		self.count_fn = count_fn

	def filter(self, val):
		self.filter_on = val

tools = dict()
tools['general'] = general
tools['positional'] = positional
tools['simple_pos'] = simple_pos
tools['verb_pos'] = verb_pos
tools['nn_jj_pos'] = nn_jj_pos
tools['mcw'] = most_common_words