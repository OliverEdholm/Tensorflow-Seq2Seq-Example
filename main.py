# imports
from config import BATCH_SIZE
from config import EPOCHS
from config import LEARNING_RATE
from config import MAX_LENGTH
from data import get_batch

from six.moves import xrange

import numpy as np
import tensorflow as tf
from tensorflow.contrib.legacy_seq2seq import basic_rnn_seq2seq
from tensorflow.contrib.rnn import BasicLSTMCell


# functions
def build_model(X, y):
	y_, _ = basic_rnn_seq2seq(X, y, BasicLSTMCell(4))

	return y_	


def get_loss(y, y_):
	reshaped_outputs = tf.reshape(y_, [-1])
	reshaped_results = tf.reshape(y, [-1])

	loss = tf.losses.mean_squared_error(reshaped_outputs, reshaped_results)	

	return loss


def get_series(batch, series_idx):
	series = [batch[idx][series_idx]
	          for idx in xrange(MAX_LENGTH)]

	return np.array(series)


def convert_prediction(pred, series_idx):
	series = get_series(pred, series_idx) 

	converted_series = []
	for value in series:
		converted_value = np.zeros(len(value), dtype=np.int32)
		converted_value[value.argmax()] = 1

		converted_series.append(converted_value)

	return np.array(converted_series)


def main():
	X = [tf.placeholder('float', [BATCH_SIZE, 4])
		 for _ in xrange(MAX_LENGTH)]
	y = [tf.placeholder('float', [BATCH_SIZE, 4])
	     for _ in xrange(MAX_LENGTH)]
	
	y_ = build_model(X, y)

	loss = get_loss(y, y_)

	optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
	train_operation = optimizer.minimize(loss)

	init = tf.global_variables_initializer()	
	with tf.Session() as sess:
		sess.run(init)

		for epoch in xrange(EPOCHS):
			X_batch, y_batch = get_batch()

			X_list = dict(zip(X, X_batch))
			y_list = dict(zip(y, y_batch))

			merged_dict = {**X_list, **y_list}
			loss_val, pred, _  = sess.run([loss, y_, train_operation],
			    						  feed_dict=merged_dict)

			if (epoch + 1) % 1000 == 0:
				print('epoch: {}, loss: {}'.format(epoch + 1, loss_val))
				print('Input:')
				print(get_series(X_batch, 0))
				print()
				print('Ground truth:')
				print(get_series(y_batch, 0))
				print()
				print('Prediction:')
				print(convert_prediction(pred, 0))
				print()
				print()


if __name__ == '__main__':
	main()
