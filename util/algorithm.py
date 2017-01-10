import math
import random as rd


def argmax(arr):
    if isinstance(arr, list):
        return arr.index(max(arr))
    if isinstance(arr, Matrix):
        return [argmax(l) for l in arr]


def int_to_list(integer, length):
    """
    use a list with 0 and 1 to represent a non-negative integer
    :param integer: the integer to be converted
    :param length: the length of expected list
    :return: a list representing the integer
    """
    assert length > integer
    temp = [0 for i in range(length)]
    temp[integer] = 1
    return temp


def exp(x):
    if isinstance(x, Matrix):
        return Matrix([[math.e ** x[row][col] for col in range(x.shape[1])] for row in range(x.shape[0])])
    return math.e ** x


def sigmoid(x, lb=0, ub=1):
    """
    Sigmoid function, use lower bound and upper bound to convert data to any scale
    :param x:
    :param lb: lower bound
    :param ub: upper bound
    :return: result
    """

    # since image data is too large, use approximate value to speed up program
    if isinstance(x, Matrix):
        arr = [[0 for col in range(x.shape[1])] for row in range(x.shape[0])]
        for row in range(x.shape[0]):
            for col in range(x.shape[1]):
                if x[row][col] > 5:
                    arr[row][col] = ub
                elif x[row][col] < -5:
                    arr[row][col] = lb
                else:
                    arr[row][col] = (1. / (1. + exp(-x[row][col]))) * (ub - lb) + lb
        return Matrix(arr)
    return (1. / (1. + exp(-x))) * (ub - lb) + lb


def derive_sig(y, lb=0., ub=1.):
    # return sigmoid(x) * (1 - sigmoid(x))
    # use y to replace sigmoid(x)
    temp = (y - lb) / (ub - lb)
    return temp * (1. - temp) * (ub - lb)


def sin(x):
    return Matrix([[math.sin(x[row][col]) for col in range(x.shape[1])] for row in range(x.shape[0])])


def avg(arr):
    return sum(arr) / len(arr)


def variance(arr):
    average = avg(arr)
    return (sum(map(lambda x: (x - average) ** 2, arr))) / len(arr)


def normalize(arr):
    return list(map(lambda x: (x - min(arr)) / (max(arr) - min(arr)), arr))


class Matrix(object):
    def __init__(self, matrix):
        if isinstance(matrix, Matrix):
            self._matrix = matrix._matrix
            self.shape = matrix.shape
            return
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

    def __truediv__(self, other):
        if isinstance(other, Matrix):
            raise TypeError('Unsupported operand type for /: "Matrix" and "Matrix"')
        return Matrix(
            [[self._matrix[row][col] / other for col in range(self.shape[1])] for row in range(self.shape[0])])

    def __str__(self):
        s = ''
        for x in self._matrix:
            for y in x:
                s += str(y) + " "
            s = s[:-1] + "\n"
        return s

    def __repr__(self):
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

    def concatenate(self, other, axis=0):
        m = [[self._matrix[r][c] for c in range(self.shape[1])] for r in range(self.shape[0])]
        if not axis:
            if self.shape[0] != other.shape[0]:
                raise ValueError('Two matrix must have same row number')
            for i in range(self.shape[0]):
                m[i].extend(other._matrix[i])
            return Matrix(m)
        if self.shape[1] != other.shape[1]:
            raise ValueError('Two matrix must have same col number')
        return Matrix([self._matrix[:].extend(other._matrix[:])])

    def add_bias(self):
        return self.concatenate(Matrix.ones(self.shape[0], 1))

    @staticmethod
    def ones(row, col):
        return Matrix([[1 for c in range(col)] for r in range(row)])

    @staticmethod
    def zeros(row, col):
        return Matrix([[0 for c in range(col)] for r in range(row)])

    @staticmethod
    def random(row, col, lb=-1, ub=1):
        rd.seed(33)
        return Matrix([[rd.uniform(lb, ub) for c in range(col)] for r in range(row)])
