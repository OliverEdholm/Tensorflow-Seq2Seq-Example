# imports
from config import MIN_LENGTH
from config import MAX_LENGTH
from config import BATCH_SIZE

from random import randint
from random import choice
from six.moves import xrange

import numpy as np
import tensorflow as tf


# functions
def generate_data(min_length=MIN_LENGTH, max_length=MAX_LENGTH):
	'''
	The input concists of a series of ones and zeros. The output is the
	input where the ones and zeros gets inverted.
	Example:
	Input:  0 1 1 0 0 1
	Output: 1 0 0 1 1 0
	'''
	def add_padding(l, max_length):
		to_add = max_length - len(l)
		for _ in xrange(to_add):
			l.append([1, 0, 0, 0])

		return l

	length = randint(min_length, max_length - 1)
	X = [choice([[0, 0, 0, 1], [0, 0, 1, 0]])
	     for _ in xrange(length)]

	y = [[0, 0, 1, 0] if val == [0, 0, 0, 1] else [0, 0, 0, 1]
	     for val in X]
	y.insert(0, [0, 1, 0, 0])

	X = add_padding(X, max_length)
	y = add_padding(y, max_length)

	return np.array(X), np.array(y)


def get_batch(batch_size=BATCH_SIZE):
	X_batch = [[] for _ in xrange(MAX_LENGTH)]
	y_batch = [[] for _ in xrange(MAX_LENGTH)]

	for _ in xrange(batch_size):
		X, y = generate_data()

		for idx in xrange(MAX_LENGTH):
			X_batch[idx].append(X[idx])
			y_batch[idx].append(y[idx])

	return np.array(X_batch), np.array(y_batch)
