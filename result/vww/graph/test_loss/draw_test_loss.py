import os
import sys
sys.path.append('../')

import matplotlib.pyplot as plt

from get_info import *

plt.figure(figsize=(8, 5), dpi=150)

# draw original model
test_loss = get_test_loss('../../original/150epochs_01.txt')
plt.plot(test_loss, label='FastGRNN+Front', linestyle='-', color='gold')

# draw modified model
PREFIX, NUMBER = '150epochs', '01'
RNN_CELL_ABBR = ['fg', 'gru', 'lstm']
RNN_CELL = ['FastGRNN', 'GRU', 'LSTM']
POS_ABBR = ['front', 'last', 'fl']
POS = ['Front', 'Last', 'FrontLast']
LINE_COLOR = ['lime', 'cyan', 'green', 'blue', 'olive', 'purple', 'r', 'pink']
num = 0

for i, x in enumerate(RNN_CELL_ABBR):
    for j, y in enumerate(POS_ABBR):
        # ignore original model
        if x == 'fg' and y == 'front':
            continue

        file_name = f'{PREFIX}_{x}_{y}_{NUMBER}.txt'
        if not os.path.exists(f'../../modified/{file_name}'):
            continue
        
        test_loss = get_test_loss(f'../../modified/{file_name}')
        plt.plot(test_loss, linestyle='-', color=LINE_COLOR[num], label=f'{RNN_CELL[i]}+{POS[j]}')

        num += 1

# draw some information
plt.title('Compare Loss for 9 Models in Testing')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='best')
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(20))  # set x-axis step

plt.show()
