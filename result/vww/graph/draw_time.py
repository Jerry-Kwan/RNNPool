import numpy as np
import matplotlib.pyplot as plt

from get_info import *

epoch_time = []
label = []

# original model
epoch_time.append(np.mean(get_epoch_time('../original/150epochs_02.txt')))
label.append('FastGRNN+Front')

# modified models
PREFIX, NUMBER = '150epochs', '01'
RNN_CELL_ABBR = ['fg', 'gru', 'lstm']
RNN_CELL = ['FastGRNN', 'GRU', 'LSTM']
POS_ABBR = ['front', 'last', 'fl']
POS = ['Front', 'Last', 'FrontLast']

for i, x in enumerate(RNN_CELL_ABBR):
    for j, y in enumerate(POS_ABBR):
        if x == 'fg' and y == 'front':
            continue

        file_name = f'{PREFIX}_{x}_{y}_{NUMBER}.txt'
        epoch_time.append(np.mean(get_epoch_time(f'../modified/{file_name}')))
        label.append(f'{RNN_CELL[i]}+{POS[j]}')

ret = sorted(list(zip(epoch_time, label)), key=lambda x: x[0])
epoch_time = [x[0] for x in ret]
label = [x[1] for x in ret]

plt.figure(figsize=(16, 5), dpi=100)
barh = plt.barh(range(len(epoch_time)), epoch_time, tick_label=label)
plt.bar_label(barh, label_type='edge')
plt.xlabel('Average Time per Training Epoch (s)')
plt.ylabel('Model Name')
plt.title('Compare Average Time per Training Epoch for 9 Models')
plt.show()
