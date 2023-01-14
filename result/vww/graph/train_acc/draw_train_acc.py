import os
import sys
sys.path.append('../')

import matplotlib.pyplot as plt

from get_info import *

plt.figure(figsize=(8, 5), dpi=150)

# draw original model
train_acc = get_train_acc('../../original/150epochs_01.txt')
plt.plot(train_acc, label='FastGRNN+Front', linestyle='-', color='cyan')

# draw modified model
PREFIX, NUMBER = '150epochs', '01'
RNN_CELL_ABBR = ['fg', 'gru', 'lstm']
RNN_CELL = ['FastGRNN', 'GRU', 'LSTM']
POS_ABBR = ['front', 'last', 'fl']
POS = ['Front', 'Last', 'FrontLast']
LINE_COLOR = ['lime', 'pink', 'green', 'blue', 'olive', 'purple', 'r', 'gold']
num = 0

for i, x in enumerate(RNN_CELL_ABBR):
    for j, y in enumerate(POS_ABBR):
        # ignore original model
        if x == 'fg' and y == 'front':
            continue

        file_name = f'{PREFIX}_{x}_{y}_{NUMBER}.txt'
        if not os.path.exists(f'../../modified/{file_name}'):
            continue

        train_acc = get_train_acc(f'../../modified/{file_name}')
        plt.plot(train_acc, linestyle='-', color=LINE_COLOR[num], label=f'{RNN_CELL[i]}+{POS[j]}')

        num += 1

# draw some information
plt.title('Compare Acc for 9 Models in Training')
plt.xlabel('epoch')
plt.ylabel('acc')
plt.legend(loc='best')
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(20))  # set x-axis step

plt.show()
