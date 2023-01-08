import os
import sys
sys.path.append('../')

import matplotlib.pyplot as plt

from get_info import *

# draw original model
test_loss = get_test_loss('../../original/150epochs_01.txt')
plt.plot(test_loss, label='FastGRNN+Front', linestyle='-', color='cyan')

# draw modified model
PREFIX, NUMBER = '150epochs', '01'
RNN_CELL_ABBR = ['fg', 'gru', 'lstm']
RNN_CELL = ['FastGRNN', 'GRU', 'LSTM']
POS_ABBR = ['front', 'last', 'fl']
POS = ['Front', 'Last', 'FrontLast']
LINE_STYLE = ['-', '--', ':']
LINE_COLOR = ['cyan', 'r', 'olive']

for i, x in enumerate(RNN_CELL_ABBR):
    for j, y in enumerate(POS_ABBR):
        if x == 'fg' and y == 'front':
            continue

        file_name = f'{PREFIX}_{x}_{y}_{NUMBER}.txt'
        if not os.path.exists(f'../../modified/{file_name}'):
            continue
        
        test_loss = get_test_loss(f'../../modified/{file_name}')
        plt.plot(test_loss, linestyle=LINE_STYLE[i], color=LINE_COLOR[j], label=f'{RNN_CELL[i]}+{POS[j]}')

# some information
plt.title('Compare Loss for 9 Models in Testing')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='best')
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(20))  # xaxis step

plt.show()
