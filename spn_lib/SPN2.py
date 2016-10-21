import tensorflow as tf
import numpy as np
from model import *
from data import *

class SPN:
	def __init__(self):
		self.model = None
		self.data = None
		self.loss = []
		self.val_loss = None
		self.test_loss = None
		self.session_on = False

	def make_model_from_file(self, fname, random_weights=False):
		self.model = Model(fname, random_weights)
		self.data = Data(self.model.input_order)

	def add_data(self, filename, dataset='train'):
		if dataset == 'train':
			self.data.load_and_process_train_data(filename)
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

	def evaluate(self, inp):
		feed_dict = {self.model.input: inp}
		loss = self.model.session.run(self.model.loss, feed_dict=feed_dict)
		return loss

	def train(self, epochs, data=[], minibatch_size=512):
		if data == []:
			data = self.data.train
			print data.shape

		for e in xrange(epochs):
			print 'Epoch ' + str(e)
			for m in xrange(data.shape[1]//minibatch_size+1):
				feed_dict = {self.model.input: data[:, m*minibatch_size:min(data.shape[1], (m+1)*500)]}
				_, loss, result = self.model.session.run([self.model.opt_val, 
														  self.model.loss,
														  self.model.output], 
														  feed_dict=feed_dict)
				self.loss.append(loss)
				self.model.get_normal_value()
				# print self.model.norm_value
				print "Loss: " + str(loss)
				# print "Min: " + str(result)
				# print list(data[:, m*500:min(data.shape[1], (m+1)*500)])
				# print map(lambda x: self.model.session.run(x), self.model.sparse_tensors)
