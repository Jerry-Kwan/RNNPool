import os
import sys
sys.path.append('../')

import matplotlib.pyplot as plt

from get_info import *

path = '../../original/150epochs_01.txt'
train_loss = get_train_loss(path)
test_loss = get_test_loss(path)
plt.plot(train_loss, label='train_loss', linestyle='-', color='cyan')
plt.plot(test_loss, label='test_loss', linestyle='-', color='r')

plt.title('Train Loss and Test Loss of original model')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='best')
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(20))  # xaxis step

plt.show()
