import sys
import matplotlib.pyplot as plt

from docopt import docopt

def main():
    args = docopt("""Visualize the training outputs of a model.

    Usage:
        visualize_leanring.py <values.traininfo> [--skip=<skip>]

        <values.traininfo> = gets output after training
        --skip=<skip> = skip the first elements
    """)

    file = args['<values.traininfo>']
    skip = args['--skip']
    with open (file) as f:
        content = f.readlines()

    name = content[0].strip()
    time = content[1].strip()

    amounts = [int(val) for val in content[2].strip().split(' ')]
    acc_train = [float(val) for val in content[3].strip().split(' ')]
    acc_dev = [float(val) for val in content[4].strip().split(' ')]
    mean_loss = [float(val) for val in content[5].strip().split(' ')]

    if skip is not None:
        print('Skipping the first', skip, 'elements.')
        name += 'skip_'+skip
        skip = int(skip)

        cut_at = len(amounts)
        for i, val in enumerate(amounts):
            if val >= skip:
                cut_at = i
                break
        amounts = amounts[cut_at:]
        acc_train = acc_train[cut_at:]
        acc_dev = acc_dev[cut_at:]
        mean_loss = mean_loss[cut_at:]


    plot_learning(name, amounts, acc_train, acc_dev, mean_loss)


def plot_learning(name, amount_data, acc_train, acc_dev, mean_loss):
    plt.plot(amount_data, acc_dev,label='dev set (accuracy)')
    plt.plot(amount_data, acc_train, label='train set (accuracy)')
    plt.plot(amount_data, mean_loss, label='mean loss on train')
    plt.xlabel('# samples')
    plt.ylabel('acccuracy/loss')
    plt.legend()
    plt.title(name)
    plt.savefig('./plots/' + name +'.png')
    plt.clf()

if __name__ == '__main__':
    main()
