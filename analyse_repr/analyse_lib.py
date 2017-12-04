import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

import random
import jenkspy

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
'#ffff00',
'#00ffff'
]

def adjust_plot_size(width=30, height=30):
	fig_size = plt.rcParams["figure.figsize"]
	fig_size[0] = width
	fig_size[1] = height
	plt.rcParams["figure.figsize"] = fig_size

def general(a_set, q=None, save=None, filter_q=None, params=None, show_stats=None):
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

	sent_info = [(sidx, a_set.get(sidx, repr_idx)) for sidx in sent_idx]
	colored_sents = [settings.color_sent(w,pos,parse,a,r,repr_idx, s_idx) for s_idx, (w,pos,parse,a,r) in sent_info]
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

def plot_cluster(a_set, settings, title, save, num_clusters, labels_clusters, dim):

	if dim == None:
		print('Must specify dimension via --details <dim>')
		return -1


	def cluster(num, data):
		random.seed(1)	# make it reproducable
		breaks = jenkspy.jenks_breaks([val for val, lbl, word in data], nb_class=num)
		# since values are exact matches but classes are checked with <
		breaks[-1] += 1
		clusters = []
		for i in range(len(breaks) - 1):
			clusters.append([(v,l,w) for v,l,w in data if v >= breaks[i] and v < breaks[i+1]])
		return clusters
		
	num_clusters = int(num_clusters)
	labels = sorted(list(settings.colors().keys()))
	colors = [settings.colors()[k] for k in labels]

	if labels_clusters == None:
		labels_clusters = ['cat' + str(i+1) for i in range(num_clusters)]
	else:
		labels_clusters = labels_clusters.split(' ')

	filter_fn = None
	if settings.filter_on:
		filter_fn = settings.filter_fn

	title = title + ' (cluster=' + str(num_clusters) + ') dim=' + str(dim) + ', sd-rank=' + str(a_set.dimension_rank(dim))

	working_sent_idx = a_set.enum_sents(filter_fn)
	dim_act, dim_repr = a_set.get_dim(working_sent_idx, dim)

	activated_words = a_set.get_word_along_dim(working_sent_idx, dim_act)
	labeled_data = [(dim_repr[i], settings.label_point(dim, sent_idx, dim_act[i]), activated_words[i]) for i, sent_idx in enumerate(working_sent_idx)]

	# sort and cluser
	clusters = cluster(num_clusters, labeled_data)
	
	def print_clusters(clusters, labels_clusters):
		# print per cluster: each category
		print('By Cluster: by label')
		print('====================')
		for i, cluster in enumerate(clusters):
			print(labels_clusters[i] + ':')
			for lbl in labels:
				print('* ' + lbl + ':')
				print(Counter([w for v,l,w in cluster if l == lbl]))
				print('')

		# get word lists and conflicts	
		word_counters = []
		for i, cluster in enumerate(clusters):
			word_counters.append(Counter([w for v,l,w in cluster]))

		# find overlaps
		vocab = set([w for wc in word_counters for w in wc.keys()])
		conflicts = [[] for i in range(len(labels_clusters))]
		for w in vocab:
			containers = []
			for i,wc in enumerate(word_counters):
				if w in wc:
					containers.append((i, wc, wc[w]))
			if len(containers) > 1:
				# remove from words, but keep as suggestion
				cat_most_likely = sorted(containers, key=lambda x: -x[-1])[0][0]
				conflicts[cat_most_likely].append(w + ': ' + '; '.join([labels_clusters[i] + '=' + str(wc[w]) for i, wc, _ in containers]))
				for i, wc,_ in containers:
					word_counters[i].pop(w, None)

		print('Word lists')
		print('==========')
		for i in range(len(labels_clusters)):
			print(labels_clusters[i] + '=')
			print(' '.join([w for w,amount in word_counters[i].most_common()]))
			print('Conflicts:')
			for conflict in conflicts[i]:
				print(conflict)
		

	def plot_distribution(labels_clusters):
		plt.figure(200)
		y_offset = np.zeros((len(clusters),), dtype=np.int)
		x_axis = np.arange(len(clusters))
		width = 0.35

		plts = []
		plt_lbls = [] 
		for lbl in labels:
			amounts = np.array([len([(v,l) for v,l,w in c if l == lbl]) for c in clusters])	
			p = plt.bar(x_axis, amounts, width, color=settings.colors()[lbl], bottom=y_offset)
			y_offset += amounts

			plts.append(p[0])
			plt_lbls.append(lbl)

		plt.ylabel('# sentences (' + str(len(working_sent_idx)) + ' / ' + str(len(a_set.sents)) + ')')
		plt_title = title + ' (distributional)'
		plt.title(plt_title)			
		plt.xticks(x_axis, labels_clusters)
		plt.legend(plts, plt_lbls,loc='center left', bbox_to_anchor=(1, 0.5))
		plt.subplots_adjust(right=0.8)

		if save:
			plt.savefig('./plots/' + plt_title +'.png')

		plt.show(block=False)

	def plot_stats(labels_clusters):
		plt.figure(300)
		labels_clusters += ['[DIM=' + str(dim) + ']']
		x_axis = np.arange(len(labels_clusters))

		#mean = np.mean(r)
		#sd = np.std(r)
		#abs_min = np.amin(r)
		#abs_max = np.amax(r)

		cluster_vals = [[v for v,l,w in c] for c in clusters]

		mean = [np.mean(c) for c in cluster_vals] + [a_set.mean[dim]]
		sd = [np.std(c) for c in cluster_vals] + [a_set.sd[dim]]
		vmin = [np.amin(c) for c in cluster_vals] + [a_set.min[dim]]
		vmax = [np.amax(c) for c in cluster_vals] + [a_set.max[dim]]

		plt_mean = plt.errorbar(x_axis, mean, sd, linestyle='None', marker='.', color='green', ecolor='red', elinewidth=1, markersize=5, alpha=1)
		plt_min = plt.scatter(x_axis, vmin, s=5, alpha=1)
		plt_max = plt.scatter(x_axis, vmax, s=5, alpha=1)

		plt.legend([plt_mean, plt_min, plt_max], ['mean with sd', 'min values', 'max values'], bbox_to_anchor=(0,1.1,1,0.2), loc="lower left",mode="expand", borderaxespad=0, ncol=3)
		plt.subplots_adjust(top=0.8)
		plt.xticks(x_axis, labels_clusters)
		plt.xlabel('categories')
		plt.ylabel('value of representation')
		plt_title = title + ' (statistics)'
		plt.title(plt_title)

		adjust_plot_size(width=9, height=6)
		if save:
			plt.savefig('./plots/' + plt_title +'.png')

		plt.show()

	print_clusters(clusters, labels_clusters)
	plot_distribution(labels_clusters)
	plot_stats(labels_clusters)

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

def words_analysis(a_set, name, word_fn, labels, exclude=None, dim=None, save=False, q=None, filter_q=None, show_stats=False, ext_priority_fn=None, params=None):
	
	if filter_q != None:
		filter_q = filter_q.split(' ')

	num_clusters = params['num_clusters']
	lbl_clusters = params['cluster_labels']

	def colors():
		''' Map category to color '''
		colors = {lbl : color_palette[i % len(color_palette)] for i, lbl in enumerate(labels)}
		return colors

	def color_sent(words, pos, parse, act, repr, indizes, sent_idx):
		''' Create dict() for sent with k=category, v=[indizes] '''
		result = dict()
		for lbl in labels:
			result[lbl] = [idx for i, idx in enumerate(indizes) if word_fn(words[act[i]]) == lbl]
		return result

	def distribution(sent_indizes, dim):
		''' Create dict() for activations with k=category, v=len(activations) '''
		act, _ = a_set.get_dim(sent_indizes, dim)
		words = a_set.get_word_along_dim(sent_indizes, act)
		word_categories = [word_fn(w) for w in words]
		
		result = dict()
		for lbl in labels:
			result[lbl] = [(word_cat, words[i]) for i,word_cat in enumerate(word_categories) if word_cat == lbl]

		# print words
		print_dist(words, result)

		for k in result.keys():
			result[k] = len(result[k])

		return result

	def stats(dim, activations, representations, sent_indizes):
		'''Create dict with k=category, v=(mean, sd, min, max)'''
		result = dict()
		word_categories = [word_fn(w) for w in a_set.get_word_along_dim(sent_indizes, activations)]

		# remember repr values per category
		for lbl in labels:
			result[lbl] = [representations[i] for i in range(len(representations)) if lbl == word_categories[i]]

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
		word_categories = [word_fn(w) for w in a_set.get_word_along_dim(sent_indizes, activations)]	
		if exclude != None:
			word_categories = [w for w in word_categories if w != exclude]	

		most_common = Counter(word_categories).most_common(1)
		if len(most_common) > 0:
			_, most_freq = most_common[0]
			return most_freq
		else:
			return 0

	def query_fn(sent_indizes, activations):
		'''return a score for the dimension for containing the query'''
		word_categories = [word_fn(w) for w in a_set.get_word_along_dim(sent_indizes, activations)]		
		return len([w for w in word_categories if w == q])

	def filter_fn(sent_idx):
		sent_words = [word_fn(w) for w in a_set.sents[sent_idx]]
		for filter_word in filter_q:
			if filter_word in sent_words:
				return True

		return False 

	def count_fn(working_sent_idx, labels):
		'''Count Occurences for each'''
		word_categories = [word_fn(a_set.sents[sent_idx][w_idx]) for sent_idx in working_sent_idx for w_idx in range(len(a_set.sents[sent_idx]))]
		counter = Counter(word_categories)
		return counter

	def label_point(dim, sent_idx, act):
		#pos, sent_idx, act=None
		word = a_set.sents[sent_idx][act]
		return word_fn(word)


	settings = GridSettings(color_sent, colors, distribution, stats, label_point, filter_fn, count_fn)
	if filter_q != None:
		settings.filter(True)
		name +='[filter=' + str(filter_q) + ']'

	if num_clusters != None:
		plot_cluster(a_set, settings, name, save, num_clusters, lbl_clusters, dim)
	elif ext_priority_fn != None:
		plot_grid(a_set, ext_priority_fn, name + ' of 300 dimensions (most activations for group)', settings)
	elif show_stats:
		plot_general_stats(a_set, name + ' Overview', settings, save)
	elif dim == None and q == None:
		plot_grid(a_set, 'positional',name + ' of 300 dimensions (first dimensions)', settings)
		plot_grid(a_set, 'sd', name + ' of 300 dimensions (most SD)', settings)
		plot_grid(a_set, priority_fn, name + ' of 300 dimensions (most activations per single position)', settings)
	elif q != None:	
		plot_grid(a_set, query_fn, name + ' of 300 dimensions (most of:' + q + ')', settings)
	else:
		plot_dim_details(a_set, settings, dim, name, save)

def words(a_set, dim=None, save=False, q=None, filter_q=None, show_stats=False, params=None):

	w_list = params['w_list']
	g_list = params['g_list']
	grouping = params['group']

	w_fn = None 		# word function
	g_fn = None 	# grouping function
	name = None

	if w_list != None:	
		name = 'Lexical Analysis: ' + w_list
		labels = w_list.split(' ')
		labels.append('OTHER')

		def word_fn(w):
			if w.lower() in labels:
				return w.lower()
			else:
			 return 'OTHER'

		def grouping_fn(sent_indizes, activations):
			'''return a score for the dimension for having other category than OTHER'''
			word_categories = [word_fn(w) for w in a_set.get_word_along_dim(sent_indizes, activations)]
			return len([w for w in word_categories if w != 'OTHER'])

		w_fn = word_fn
		g_fn = grouping_fn

	else:

		# get labels
		categories = g_list.split(';')
		categories = [cat.split('=') for cat in categories]
		categories = [(cat[0].strip(), [w.strip().lower() for w in cat[1].split(' ')]) for cat in categories]

		labels = [lbl for lbl, words in categories]
		all_words = [w for lbl, words in categories for w in words]
		labels.append('OTHER')

		name = 'Lexical Analysis: ' + ' '.join(labels)

		def word_fn(w):
			w = w.lower()
			for lbl, words in categories:
				if w in words:
					return lbl
			return 'OTHER'

		def grouping_fn(sent_indizes, activations):
			mapped_words = [word_fn(w) for w in a_set.get_word_along_dim(sent_indizes, activations)]
			return len([w for w in mapped_words if w != 'OTHER'])

		w_fn = word_fn
		g_fn = grouping_fn

	priority_fn = None
	if grouping:
		# change priority fn
		priority_fn = g_fn
		name += '(grouping)'

	words_analysis(a_set, name, w_fn, labels, exclude='OTHER', dim=dim, save=save, q=q, filter_q=filter_q, show_stats=show_stats, ext_priority_fn=priority_fn, params=params)
		
def most_common_words(a_set, dim=None, save=False, q=None, filter_q=None, show_stats=False, params=None):
	

	# find most common words
	all_words = [a_set.sents[sent_idx][w_idx] for sent_idx in range(len(a_set.sents)) for w_idx in range(len(a_set.sents[sent_idx]))]
	most_common = Counter(all_words).most_common(13)
	labels = [lbl for lbl, cnt in most_common if lbl != '.']	# . already coverd in POS
	labels.append('OTHER')

	def word_fn(w):
		if w in labels:
			return w
		else:
			return 'OTHER'
	

	name = 'Most common words'
	words_analysis(a_set, name, word_fn, labels, exclude='OTHER', dim=dim, save=save, q=q, filter_q=filter_q, show_stats=show_stats, params=params)

def pos_fn(a_set, name, pos_fn, exclude=None, dim=None, save=False, q=None, filter_q=None, show_stats=False, labels=None, params=None):
	
	num_clusters = params['num_clusters']
	lbl_clusters = params['cluster_labels']

	if filter_q != None:
		filter_q = filter_q.split(' ')

	if labels == None:
		labels = sorted(list(set([pos_fn(p) for ps in a_set.pos for p in ps])))

	def colors():
		''' Map category to color '''
		colors = {lbl : color_palette[i] for i, lbl in enumerate(labels)}
		return colors

	def color_sent(words, pos, parse, act, repr, indizes, sent_idx):
		''' Create dict() for sent with k=category, v=[indizes] '''
		result = dict()
		for lbl in labels:
			result[lbl] = [idx for i, idx in enumerate(indizes) if pos_fn(pos[act[i]], sent_idx, act[i]) == lbl]
		return result

	def distribution(sent_indizes, dim):
		''' Create dict() for activations with k=category, v=len(activations) '''
		act, _ = a_set.get_dim(sent_indizes, dim)
		words = a_set.get_word_along_dim(sent_indizes, act)
		pos = [pos_fn(p, sent_indizes[i], act[i]) for i, p in enumerate(a_set.get_pos_along_dim(sent_indizes, act))]
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
		pos = [pos_fn(p, sent_indizes[i], activations[i]) for i,p in enumerate(a_set.get_pos_along_dim(sent_indizes, activations))]

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
		pos = [pos_fn(p, sent_indizes[i], activations[i]) for i,p in enumerate(a_set.get_pos_along_dim(sent_indizes, activations))]	
		if exclude != None:
			pos = [p for p in pos if p != exclude]	

		_, most_freq = Counter(pos).most_common(1)[0]
		return most_freq

	def query_fn(sent_indizes, activations):
		'''return a score for the dimension for containing the query'''
		pos = [pos_fn(p, sent_indizes[i], activations[i]) for i,p in enumerate(a_set.get_pos_along_dim(sent_indizes, activations))]		
		return len([p for p in pos if p == q])

	def filter_fn(sent_idx):
		pos_sent = [pos_fn(p, sent_idx, act=p_idx) for p_idx, p in enumerate(a_set.pos[sent_idx])]
		for filter_pos in filter_q:
			if filter_pos in pos_sent:
				return True
		return False

	def count_fn(working_sent_idx, labels):
		'''Count Occurences for each'''
		all_pos = [pos_fn(a_set.pos[sent_idx][pos_idx], sent_idx, pos_idx) for sent_idx in working_sent_idx for pos_idx in range(len(a_set.pos[sent_idx]))]
		counter = Counter(all_pos)
		return counter

	def label_point(dim, sent_idx, act):
		#pos, sent_idx, act=None
		pos = a_set.pos[sent_idx][act]
		return pos_fn(pos, sent_idx, act)

	settings = GridSettings(color_sent, colors, distribution, stats, label_point, filter_fn, count_fn)
	if filter_q != None:
		settings.filter(True)
		name +='[filter=' + str(filter_q) + ']'

	if num_clusters != None:
		plot_cluster(a_set, settings, name, save, num_clusters, lbl_clusters, dim)
	elif show_stats:
		plot_general_stats(a_set, name + ' Overview', settings, save)
	elif dim == None and q == None:
		plot_grid(a_set, 'positional',name + ' of 300 dimensions (first dimensions)', settings)
		plot_grid(a_set, 'sd', name + ' of 300 dimensions (most SD)', settings)
		plot_grid(a_set, priority_fn, name + ' of 300 dimensions (most activations per single position)', settings)
	elif q != None:	
		plot_grid(a_set, query_fn, name + ' of 300 dimensions (most of:' + q + ')', settings)
	else:
		plot_dim_details(a_set, settings, dim, name, save)

def pos_pattern_analysis(a_set, dim=None, save=False, q=None, filter_q=None, show_stats=False, params=None):

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
		
	pos_pattern = params['pos_pattern']
	pos_pattern = [p.split('|') for p in pos_pattern.split(' ')]
	
	# labels
	counter = dict()
	labels = []
	for position in pos_pattern:
		for pos in position:
			if pos in counter:
				counter[pos] += 1
			else:
				counter[pos] = 1
			labels.append(pos + str(counter[pos]))

	labels = [l + '-first' for l in labels] + [l + '-inside' for l in labels] + [l + '-last' for l in labels] + [l + '-only' for l in labels]
	labels.append('OTHER')
	
	def find_potential_matches(pos_seq):
		matches = []
		for i in range(len(pos_seq) - len(pos_pattern)):

			found = True
			lbl_dict = dict()
			for j, pos in enumerate(pos_seq[i:i+len(pos_pattern)]):
				if pos not in pos_pattern[j]:
					found = False
					break

				potential_labels_before = [p for p_candidates in pos_pattern[:i] for p in p_candidates]
				p_idx = i+j
				lbl_dict[p_idx] = pos_seq[p_idx] + str(Counter(potential_labels_before)[pos_seq[p_idx]])


			if found:
				matches.append((i, i+len(pos_pattern), lbl_dict))

		return matches

	def to_label(pos, sent_idx, act=None):
		matches_in_sent = find_potential_matches([simplify(p) for p in a_set.pos[sent_idx]])
		pos = simplify(pos)
		
		for i, (start, end, lbl_dict) in enumerate(matches_in_sent):
			if act >= start and act < end:
				# this match
				lbl = lbl_dict[act]
				if len(matches_in_sent) == 1:
					return lbl + '-only'
				elif i == 0:
					return lbl + '-first'
				elif i == len(matches_in_sent) - 1:
					return lbl + '-last'
				else:
					return lbl + '-inside'

		# no match found
		return 'OTHER'




	name = 'Positional POS pattern "' + params['pos_pattern'] + '"'
	pos_fn(a_set, name, to_label, exclude='OTHER', dim=dim, save=save, q=q, filter_q=filter_q, show_stats=show_stats, labels=labels, params=params)


def simple_pos(a_set, dim=None, save=False, q=None, filter_q=None, show_stats=False, params=None):

	def simplify(pos, sent_idx=None, act=None):
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
	pos_fn(a_set, name, simplify, exclude=None, dim=dim, save=save, q=q, filter_q=filter_q, show_stats=show_stats, params=params)

def verb_pos(a_set, dim=None, save=False, q=None, filter_q=None, show_stats=False, params=None):

	def only_verbs(pos, sent_idx=None, act=None):
		if pos.startswith('V'):
			return pos
		else:
			return 'OTHER'

	name = 'Verb POS'
	pos_fn(a_set, name, only_verbs, exclude='OTHER', dim=dim, save=save, q=q, filter_q=filter_q, show_stats=show_stats, params=params)

def nn_jj_pos(a_set, dim=None, save=False, q=None, filter_q=None, show_stats=False, params=None):

	def only_nn_jj(pos, sent_idx=None, act=None):
		if pos.startswith('NN') or pos.startswith('JJ'):
			return pos
		else:
			return 'OTHER'

	name = 'Noun - Adj POS'
	pos_fn(a_set, name, only_nn_jj, exclude='OTHER', dim=dim, save=save, q=q, filter_q=filter_q, show_stats=show_stats, params=params)
	


def positional(a_set, dim=None, save=False, q=None, filter_q=None, show_stats=False, params=None):
	labels = [i for i in range(a_set.sent_len)]
	num_clusters = params['num_clusters']
	lbl_clusters = params['cluster_labels']

	def colors():
		''' Map category to color '''
		# 8 positions
		colors = {str(k) : color_palette[i] for i, k in enumerate(labels)}
		return colors

	def color_sent(words, pos, parse, act, repr, indizes, sent_idx):
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

	def label_point(dim, sent_idx, act):
		return str(act)

	settings = GridSettings(color_sent, colors, distribution, stats, label_point)

	if num_clusters != None:
		plot_cluster(a_set, settings, 'Word position', save, num_clusters, lbl_clusters, dim)
	elif dim == None and q == None:
		plot_grid(a_set, 'positional', 'Word position (first dimensions)', settings)
		plot_grid(a_set, 'sd', 'Word position (most SD)', settings)
		plot_grid(a_set, priority_fn, 'Word position (most activations per single position)', settings)
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
	def __init__(self, color_sent, colors, distribution, stats, label_point, filter_fn=None, count_fn=None):
		self.color_sent = color_sent
		self.colors = colors
		self.distribution = distribution
		self.stats = stats
		self.filter_fn = filter_fn
		self.filter_on = False
		self.count_fn = count_fn
		self.label_point = label_point

	def filter(self, val):
		self.filter_on = val

tools = dict()
tools['general'] = general
tools['positional'] = positional
tools['simple_pos'] = simple_pos
tools['verb_pos'] = verb_pos
tools['nn_jj_pos'] = nn_jj_pos
tools['mcw'] = most_common_words
tools['words'] = words
tools['pp'] = pos_pattern_analysis