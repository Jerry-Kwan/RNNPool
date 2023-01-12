import os
import sys
sys.path.append('../')

import matplotlib.pyplot as plt

from get_info import *

path = '../../original/150epochs_01.txt'
train_acc = get_train_acc(path)
test_acc = get_test_acc(path)
plt.plot(train_acc, label='train_acc', linestyle='-', color='cyan')
plt.plot(test_acc, label='test_acc', linestyle='-', color='r')

plt.title('Train Acc and Test Acc of original model')
plt.xlabel('epoch')
plt.ylabel('acc')
plt.legend(loc='best')
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(20))  # xaxis step

plt.show()
