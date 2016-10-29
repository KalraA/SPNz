import tensorflow as tf
import numpy as np
from model import *
from data import *

class SPN:
	def __init__(self):
		self.model = None
		self.data = None
		self.loss = []
		self.val_loss = []
		self.test_loss = None
		self.session_on = False
		self.input_order = []

	def make_model_from_file(self, fname, random_weights=False):
		self.model = Model()
		self.model.build_model_from_file(fname, random_weights)
		self.model.compile()
		self.data = Data(self.model.input_order)
		self.input_order = self.model.input_order

	def make_random_model(self, bfactor, input_size):
		self.model = Model()
		self.model.build_random_model(bfactor, input_size)
		self.model.compile()
		self.data = Data(self.model.input_order)
		self.input_order = self.model.input_order

	def add_data(self, filename, dataset='train'):
		if dataset == 'train':
			self.data.load_and_process_train_data_mem(filename)
		elif dataset == 'valid':
			self.data.load_and_process_valid_data(filename)
		elif dataset == 'test':
			self.data.load_and_process_test_data(filename)

	def start_session(self):
		self.model.start_session()
		self.session_on = True

	def close_session(self):
		self.model.close_session()
		self.session_on = False

	def predict(self, inp):
		feed_dict = {self.model.input: inp}
		output = self.model.session.run(self.model.output, feed_dict=feed_dict)
		return output

	def evaluate(self, inp, summ):
		feed_dict = {self.model.input: inp, self.model.summ: summ}
		loss, summ = self.model.session.run([self.model.loss, self.model.loss_summary], feed_dict=feed_dict)
		return loss, summ

	def train(self, epochs, data=[], minibatch_size=512, valid=True, test=True):
		if data == []:
			data = self.data.train
			print data.shape
		if (valid):
				val_loss, val_sum = self.evaluate(self.data.valid, 'valid_loss')
				self.model.writer.add_summary(val_sum, 0)

		if (test):
			test_loss, test_sum = self.evaluate(self.data.test, 'test_loss')
			self.model.writer.add_summary(test_sum, 0)

		for e in xrange(epochs):
			print 'Epoch ' + str(e)
			if e > 0:
				minibatch_size = 128
			np.random.shuffle(data)
			for m in xrange(data.shape[0]//minibatch_size+1):
				n_data = data[m*minibatch_size:min(data.shape[0], (m+1)*minibatch_size)]
				n_data = n_data[:, self.input_order, :]
				n_data = np.reshape(n_data, (len(n_data), len(self.input_order)*2))
				feed_dict = {self.model.input: n_data.T, self.model.summ: "minibatch_loss"}
				if e == 0:
					_, loss, result, summary = self.model.session.run([self.model.opt_val, 
															  self.model.loss,
															  self.model.output, 
															  self.model.loss_summary], 
															  feed_dict=feed_dict)
				else:
					_, loss, result, summary = self.model.session.run([self.model.opt_val2, 
															  self.model.loss,
															  self.model.output, 
															  self.model.loss_summary], 
															  feed_dict=feed_dict)
				self.model.writer.add_summary(summary, e*data.shape[0]//minibatch_size+1 + m)
				self.loss.append(loss)
				self.model.get_normal_value()
				# print self.model.norm_value
				# print "Loss: " + str(loss)
			if (valid):
				val_loss, val_sum = self.evaluate(self.data.valid, 'valid_loss')
				self.model.writer.add_summary(val_sum, e+1)
				print val_loss

			if (test):
				test_loss, test_sum = self.evaluate(self.data.test, 'test_loss')
				self.model.writer.add_summary(test_sum, e+1)
				# print "Min: " + str(result)
				# print list(data[:, m*500:min(data.shape[1], (m+1)*500)])
				# print map(lambda x: self.model.session.run(x), self.model.sparse_tensors)
