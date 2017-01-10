import math

from model.LCD_digit import BPNetwork
from util import io
from util.algorithm import Matrix
import pylab as pl
import sys
from util.io import load_data

SIN_TRAIN_SET_PATH = 'data/train_sin.txt'
SIN_VALIDATION_SET_PATH = 'data/validation_sin.txt'
SELECTED_WEIGHT_NAME = 'sin_weight_n=1000_r=0.50_hidden=5'


def init_input_set(n, path):
    start = -math.pi / 2
    end = math.pi / 2
    step = (end - start) / n

    set_str = ''
    for i in range(n):
        num = start + i * step
        set_str += str(num) + ' ' + str(math.sin(num)) + '\n'

    with open(path, 'w') as f:
        f.write(set_str)


if __name__ == '__main__':
    if not len(sys.argv) == 2:
        print('Usage: sine_simulator.py data_dir')
        exit(0)
    input_data = io.load_input(sys.argv[1] + '/sine.txt')
    network = BPNetwork(out_lb=-1, weight_filename=SELECTED_WEIGHT_NAME)

    predict = network.predict(input_data, model='sin', output_path='Sine[14302010033].txt')
    print(predict)

    # train and validation code

    # train_data = load_data(SIN_TRAIN_SET_PATH, 'sin')
    # train_x = Matrix(train_data[0])
    # train_y = Matrix(train_data[1])
    #
    # path = r'sin_weight_n=1000_r=0.50_hidden=5'
    # # path = None
    #
    # network = BPNetwork(train_x, train_y, hidden_num=10, out_lb=-1, out_ub=1, title='sin',
    #                     learning_rate=(0.3, 0.2), weight_filename=path)
    # # network.train(1000)
    #
    # v_data = load_data(SIN_VALIDATION_SET_PATH, model='sin')
    # v_x = Matrix(v_data[0])
    # v_y = Matrix(v_data[1])
    #
    # result = network.predict(v_x)
    #
    # compare = zip(result, v_y)
    # for r in compare:
    #     print(str(r[0]) + '\t' + str(r[1]))
    #
    # pl.title(u'n={0:d} r={1:.2f} hidden={2:d}'.format(1000, 0.5, 10))
    # pl.plot(v_data[0], v_data[1], color='blue', label='sin')
    # pl.plot(v_data[0], result.T[0], color='red', label='predict')
    # pl.legend(loc='upper left')
    # pl.show()
