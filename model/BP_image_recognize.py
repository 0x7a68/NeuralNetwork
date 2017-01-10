import sys

from model.LCD_digit import BPNetwork
from util import io

from util.algorithm import Matrix
from util.io import load_data

TRAIN_IMAGE_PATH = u'data/char/train'
VALIDATION_IMAGE_PATH = u'data/char/validation'

TRAIN_TEXT_PATH = 'data/train_image.txt'
VALIDATION_TEXT_PATH = 'data/validation_image.txt'

TEST_IMAGE_TEXT_PATH = 'data/test_image'

if __name__ == '__main__':
    if not len(sys.argv) == 3:
        print('Usage: BP_image_recognize.py data_dir image_num')
        exit(0)

    io.image_to_txt(sys.argv[1], TEST_IMAGE_TEXT_PATH, False, sys.argv[2])
    input_data = io.load_input(TEST_IMAGE_TEXT_PATH)

    network = BPNetwork(weight_filename='Image_epoch=400_hidden=10_rate0=0.008_rate1=0.001')

    predict = network.predict(input_data, model='image', output_path='LetterBP[14302010033].txt')
    print(predict)

    # train and validation code

    # image_to_txt(TRAIN_IMAGE_PATH, TRAIN_TEXT_PATH)
    # image_to_txt(VALIDATION_IMAGE_PATH, VALIDATION_TEXT_PATH)
    #
    # train_data = load_data(TRAIN_TEXT_PATH, model='image')
    # train_x = Matrix(train_data[0])
    # train_y = Matrix(train_data[1])
    #
    # network = BPNetwork(train_x, train_y, hidden_num=10, out_lb=0, out_ub=1, title='image',
    #                     learning_rate=(.5, .5))
    # # rate 0.001 too small, 0.002 too large
    # network.train(2)
    #
    # v_data = load_data(VALIDATION_TEXT_PATH, model='image')
    # v_x = Matrix(v_data[0])
    # v_y = Matrix(v_data[1])
    #
    # result = network.predict(v_x)
    #
    # compare = zip(result, v_y)
    # for r in compare:
    #     print(str(r[0]) + '\t' + str(r[1]))
