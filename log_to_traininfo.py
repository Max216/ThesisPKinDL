import sys

from docopt import docopt

def main():
    args = docopt("""Create a .traininfo for visualizing learning from the logs.

    Usage:
        log_to_traininfo.py <logfile> <trainfile>

        <logfile> = output from training
        <trainfile> = resulting .traininfo file
    """)

    # after seeing 32 samples:
	# Accuracy on train data: 0.3326810674831215
	# Accuracy on dev data: 0.3286933550091445
	# mean loss 1.098198413848877

    key_number_samples = 'after seeing'
    key_train_data = 'Accuracy on train data'
    key_dev_data = 'Accuracy on dev data'
    key_loss = 'mean loss'

    file = args['<logfile>']
    file_out = args['<trainfile>']

    amounts = []
    acc_train = []
    acc_dev = []
    mean_loss = []

    with open(file) as f_in:
        for line in f_in:
            if line.startswith(key_number_samples):
                amounts.append(line.split(' ')[2].strip())
            elif line.startswith(key_train_data):
                acc_train.append(line.split(' ')[-1].strip())
            elif line.startswith(key_dev_data):
                acc_dev.append(line.split(' ')[-1].strip())
            elif line.startswith(key_loss):
                mean_loss.append(line.split(' ')[-1].strip())
            # else ignore

    if not file_out.endswith('.traininfo'):
        file_out += '.traininfo'
    with open(file_out, 'w') as f_out:
        f_out.write(file.split('.')[0] + '\n')
        f_out.write('\n')
        f_out.write(' '.join(amounts) + '\n')
        f_out.write(' '.join(acc_train) + '\n')
        f_out.write(' '.join(acc_dev) + '\n')
        f_out.write(' '.join(mean_loss) + '\n')

    print('Done.')

if __name__ == '__main__':
    main()
