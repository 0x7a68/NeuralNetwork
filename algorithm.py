import math
import random as rd


def exp(x):
    if isinstance(x, Matrix):
        return Matrix([[math.e ** x[row][col] for col in range(x.shape[1])] for row in range(x.shape[0])])
    return math.e ** x


# sigmoid function
def sigmoid(x):
    return 1 / (1 + exp(-x))


def derive_sig(y):
    # return sigmoid(x) * (1 - sigmoid(x))
    # use y to replace sigmoid(x)
    return y * (1 - y)


def random(row, col):
    return Matrix([[rd.uniform(-1, 1) for c in range(col)] for r in range(row)])


def sin(x):
    return Matrix([[math.sin(x[row][col]) for col in range(x.shape[1])] for row in range(x.shape[0])])


class Matrix(object):
    def __init__(self, matrix: object) -> object:
        if not isinstance(matrix, list):
            raise TypeError('Expected a 2-dimensional list')
        self._matrix = matrix[:]
        row_num = len(matrix)
        col_num = set([len(x) for x in matrix])
        if len(col_num) > 1:
            raise ValueError('All lists must have the same length')
        col_num = col_num.pop()
        self.shape = (row_num, col_num)

    def __getattr__(self, item):
        if item == 'T':
            return Matrix([[self._matrix[row][col] for row in range(self.shape[0])] for col in range(self.shape[1])])
        raise AttributeError

    def __getitem__(self, index):
        return self._matrix[index]

    def __len__(self):
        return len(self._matrix)

    def __add__(self, other):
        if isinstance(other, Matrix):
            if self.shape != other.shape:
                raise ValueError('Two matrices must have the same shape')
            return Matrix([[self._matrix[x][y] + other._matrix[x][y]
                            for y in range(self.shape[1])] for x in range(self.shape[0])])
        return Matrix([[self._matrix[x][y] + other for y in range(self.shape[1])] for x in range(self.shape[0])])

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Matrix):
            if self.shape != other.shape:
                raise ValueError('Two matrices must have the same shape')
            return Matrix([[self._matrix[x][y] - other._matrix[x][y]
                            for y in range(self.shape[1])] for x in range(self.shape[0])])
        return Matrix([[self._matrix[x][y] - other for y in range(self.shape[1])] for x in range(self.shape[0])])

    def __rsub__(self, other):
        if isinstance(other, Matrix):
            if self.shape != other.shape:
                raise ValueError('Two matrices must have the same shape')
            return Matrix([[other._matrix[x][y] - self._matrix[x][y]
                            for y in range(self.shape[1])] for x in range(self.shape[0])])
        return Matrix([[other - self._matrix[x][y] for y in range(self.shape[1])] for x in range(self.shape[0])])

    def __neg__(self):
        return 0 - self

    def __mul__(self, other):
        if isinstance(other, Matrix):
            if self.shape != other.shape:
                raise ValueError('Two matrices must have the same shape')
            return Matrix([[other._matrix[x][y] * self._matrix[x][y]
                            for y in range(self.shape[1])] for x in range(self.shape[0])])
        return Matrix([[other * self._matrix[x][y] for y in range(self.shape[1])] for x in range(self.shape[0])])

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rtruediv__(self, other):
        if isinstance(other, Matrix):
            raise TypeError('Unsupported operand type for /: "Matrix" and "Matrix"')
        return Matrix(
            [[other / self._matrix[row][col] for col in range(self.shape[1])] for row in range(self.shape[0])])

    def __str__(self):
        s = '['
        for x in self._matrix:
            s += str(x) + ',\n'
        s = s[:-2] + ']'
        return s

    def dot(self, other):
        if self.shape[1] != other.shape[0]:
            raise ValueError('Shapes (%d, %d) and (%d, %d) not aligned'
                             % (self.shape[0], self.shape[1], other.shape[0], other.shape[1]))
        result = [[0 for col in range(other.shape[1])] for row in range(self.shape[0])]
        for i in range(self.shape[0]):
            for j in range(other.shape[1]):
                for k in range(self.shape[1]):
                    result[i][j] += self._matrix[i][k] * other._matrix[k][j]
        return Matrix(result)
