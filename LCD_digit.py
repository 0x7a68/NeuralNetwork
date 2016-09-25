import algorithm
from algorithm import Matrix


class BPNetwork(object):
    def __init__(self, input, output, hidden_num):
        self.weight = [0, 0]
        self.error = [0, 0]
        self.delta = [0, 0]
        self.layer = [input, 0, 0]
        self.input = input
        self.output = output

        # initialize weights of two layers
        self.weight[0] = algorithm.random(len(input[0]), hidden_num)
        self.weight[1] = algorithm.random(hidden_num, len(output[0]))

    def train(self, max_error, max_iter, learning_rate=1):
        for i in range(max_iter):
            # forward propagation
            self.layer[1] = algorithm.sigmoid(self.layer[0].dot(self.weight[0]))
            self.layer[2] = algorithm.sigmoid(self.layer[1].dot(self.weight[1]))

            # calculate error and delta of output layer
            self.error[1] = self.output - self.layer[2]
            self.delta[1] = learning_rate * self.layer[1].T.dot(self.error[1] * algorithm.derive_sig(self.layer[2]))

            # back propagation
            self.error[0] = self.error[1].dot(self.weight[1].T)
            self.delta[0] = learning_rate * self.layer[0].T.dot(self.error[0] * algorithm.derive_sig(self.layer[1]))

            # change weight
            for j in range(len(self.weight)):
                self.weight[j] += self.delta[j]
        print("Training finished")

    def predict(self, input):
        output = input
        for i in range(len(self.weight)):
            output = algorithm.sigmoid(output.dot(self.weight[i]))
        print(output)


if __name__ == '__main__':
    # input data set
    # each row represent a number from 0 to 9
    x = Matrix([[1, 1, 1, 1, 1, 1, 0],
                [0, 0, 0, 1, 1, 0, 0],
                [1, 0, 1, 1, 0, 1, 1],
                [0, 0, 1, 1, 1, 1, 1],
                [0, 1, 0, 1, 1, 0, 1],
                [0, 1, 1, 0, 1, 1, 1],
                [1, 1, 1, 0, 1, 1, 1],
                [0, 0, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 1, 1],
                [0, 1, 1, 1, 1, 1, 1]])
    y = Matrix([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

    network = BPNetwork(x, y, 5)
    network.train(0.01, 10000, 0.1)
    network.predict(x)
