import matplotlib.pyplot as plt
import numpy as np

from docopt import docopt

def load_data_to_matrix(file_path, data_type):
	with open(file_path) as f_in:
		content = [line.strip() for line in f_in.readlines() if line.startswith(data_type)]



	data = []
	for line in content:
		splitted = line.split(' ')
		amount = int(splitted[1])

		splitted_labels = splitted[0].split('-')
		gold = splitted_labels[1]
		predicted = splitted_labels[2]

		data.append((gold, predicted, amount))

	def count_items(lbl_gold, lbl_predicted):
		for g, p, a in data:
			if g == lbl_gold and p == lbl_predicted:
				return a

		return 0

	# to matrix
	labels_gold = [str(v) for v in sorted([int(v) for v in set([gold for gold, _, __ in data])])]
	labels_predicted = [str(v) for v in sorted([int(v) for v in set([predicted for gold, _, __ in data])])]

	matrix = np.zeros((len(labels_gold), len(labels_predicted)))
	for idx_gold in range(len(labels_gold)):
		for idx_predicted in range(len(labels_predicted)):
			matrix[idx_gold, idx_predicted] = count_items(labels_gold[idx_gold], labels_predicted[idx_predicted])

	return (labels_gold, labels_predicted, matrix)

def plot_confusion_matrix(matrix, labels_x, labels_y, title):
	fig, ax = plt.subplots()
	cax = ax.imshow(matrix, origin='upper')

	width, height = matrix.shape

	fig.colorbar(cax)
	plt.xticks(np.arange(len(labels_x)), labels_x, rotation=45)
	plt.yticks(np.arange(len(labels_y)), labels_y)
	plt.xlabel('Predicted')
	plt.ylabel('Gold')

	
	for x in range(width):
		for y in range(height):
			plt.annotate(str(matrix[x,y]), size=6, xy=(y, x), horizontalalignment='center', verticalalignment='center')

	plt.show()

def main():
    args = docopt("""Plot the confusion matrix of a file.

    Usage:
        result_to_conf_matrix.py  plot <path> <data>
    """)

    file_path = args['<path>']
    data_type = args['<data>']

    gold, predicted, matrix = load_data_to_matrix(file_path, data_type)
    plot_confusion_matrix(matrix, predicted, gold, 'Confusion matrix')


if __name__ == '__main__':
    main()