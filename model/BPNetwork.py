from util import algorithm
from util.algorithm import Matrix

TAG = 'tag'
FOLDER_PATH = 'data/output/'


class BPNetwork(object):
    def __init__(self, in_layer=None, out_layer=None, hidden_num=5,
                 out_lb=0, out_ub=1, weight_filename=None, title='BP', learning_rate=(1, 1)):
        """

        :param in_layer: input data set, should be a two dimensional list or a Matrix
        :param out_layer: output data set, i.e target, should be a two dimensional list or a Matrix
        :param hidden_num: hidden layer node number
        :param out_lb: output lower bound
        :param out_ub: output upper bound
        :param weight_filename: pre-trained weight filename
        :param title: a title to identify output datafile
        :param learning_rate: learning rates of two layer
        """
        self.weight = [[], []]
        self.error = [0, 0]
        self.delta = [0, 0]
        self.out_lb = out_lb
        self.out_ub = out_ub
        self.title = title
        self.learning_rate = learning_rate

        if in_layer and out_layer:
            in_layer = Matrix(in_layer).add_bias()
            out_layer = Matrix(out_layer)

            self.layer = [in_layer, 0, 0]
            self.output = out_layer
            self.hidden_num = hidden_num

            # initialize weights of two layers， +1 because of bias
            self.weight[0] = Matrix.random(len(in_layer[0]), hidden_num + 1)
            self.weight[1] = Matrix.random(hidden_num + 1, len(out_layer[0]))  # hidden_num + 1 because of bias

        if weight_filename:
            self.read_weight(weight_filename)

    def train(self, max_iter, learning_rate=None):
        if learning_rate:
            self.learning_rate = learning_rate
        for i in range(max_iter):
            # forward propagation
            l0_bias = self.layer[0]
            self.layer[1] = algorithm.sigmoid(l0_bias.dot(self.weight[0]))
            l1_bias = self.layer[1]
            self.layer[2] = algorithm.sigmoid(l1_bias.dot(self.weight[1]), self.out_lb, self.out_ub)

            # calculate error and delta of output layer
            self.error[1] = self.output - self.layer[2]
            self.delta[1] = self.learning_rate[1] * l1_bias.T.dot(
                self.error[1] * algorithm.derive_sig(self.layer[2], self.out_lb, self.out_ub))

            # back propagation
            self.error[0] = (self.error[1] * algorithm.derive_sig(self.layer[2], self.out_lb, self.out_ub)).dot(
                self.weight[1].T)
            self.delta[0] = self.learning_rate[0] * l0_bias.T.dot(
                self.error[0] * algorithm.derive_sig(self.layer[1]))

            # change weight
            for j in range(len(self.weight)):
                self.weight[j] += self.delta[j]
        print("Training finished")
        self.write_weight(
            u'{0:s}_weight_n={1:d}_r={2:.5f},{3:.5f}_hidden={4:d}'.format(self.title, max_iter, self.learning_rate[0],
                                                                          self.learning_rate[1], self.hidden_num))

    def predict(self, in_layer, model, output_path=None):
        m = in_layer
        if isinstance(in_layer, int) or isinstance(in_layer, float):
            m = Matrix([[in_layer]])
        elif isinstance(in_layer, list):
            if isinstance(in_layer[0], list):
                m = Matrix(in_layer)
            else:
                m = Matrix([in_layer])
        l1 = algorithm.sigmoid(m.add_bias().dot(self.weight[0]))
        result = algorithm.sigmoid(l1.dot(self.weight[1]), self.out_lb, self.out_ub)
        if output_path:
            with open(output_path, 'w') as f:
                if model == 'LCD':
                    for line in result:
                        f.write(str(algorithm.argmax(line)) + '\n')
                elif model == 'sin':
                    for line in result:
                        f.write(str(line[0]) + '\n')
                elif model == 'image':
                    for line in result:
                        f.write(chr(algorithm.argmax(line) + ord('A')) + '\n')
        return result

    def write_weight(self, filename):
        f = open(FOLDER_PATH + filename, 'w')
        f.write(str(self.weight[0]))
        f.write('{0}\n'.format(TAG))
        f.write(str(self.weight[1]))
        f.close()

    def read_weight(self, filename):
        f = open(FOLDER_PATH + filename)
        index = 0
        w = [[], []]
        for line in f:
            if line.startswith(TAG):
                index = 1
                continue
            w[index].append(list(map(float, line.split())))
        f.close()
        self.weight[0] = Matrix(w[0])
        self.weight[1] = Matrix(w[1])
