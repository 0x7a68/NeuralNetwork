import sys

from model.BPNetwork import BPNetwork
from util import algorithm
from util import io
from util.algorithm import Matrix


if __name__ == '__main__':
    if not len(sys.argv) == 2:
        print('Usage: LCD_digit.py data_dir')
        exit(0)
    input_data = io.load_input(sys.argv[1] + '/lcdd.txt')
    network = BPNetwork(weight_filename='LCD_weight_n=1000_r=1.00000,1.00000_hidden=5')

    predict = network.predict(input_data, model='LCD', output_path='LcdD[14302010033].txt')
    print(predict)
