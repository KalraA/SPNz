from model import Model 
from SPN2 import SPN
import numpy as np
import random
mn = 'nltcs'
minibatch_cap = 500
mem = True
valid_losses = []
test_loss = 0
my_spn = SPN()
my_spn.make_fast_model_from_file('../Modelz/' + mn + '.spncc.spn.txt', random_weights=True)
my_spn.start_session()
#my_spn.make_model_from_file('../spn_models/ + mn + .spncc.spn.txt', True)
# assert 1.1 >= my_spn.predict([[1]]*1574)[0][0]
my_spn.add_data('../Dataz/' + mn + '.ts.data', 'train', mem)
my_spn.add_data('../Dataz/' + mn + '.test.data', 'test')
my_spn.add_data('../Dataz/' + mn + '.valid.data', 'valid')
# b = my_spn.test(my_spn.data.valid.T)
# my_spn.train(20)
i = 0
# for d in my_spn.data.valid.T[:100]:
# 	i += 1
# 	my_spn.model.apply_count([d])
# 	if i % 100 == 0:
# 		print str(i) + "/" + str(len(my_spn.data.valid.T))
# vloss = my_spn.model.session.run(my_spn.model.loss, feed_dict={my_spn.model.input: my_spn.data.valid.T})
# print vloss
# valid_losses.append(vloss)
if not mem:
	data = my_spn.data.train.T
	a = 0
	ms = 2
	n = 17
	for i in range(len(data)//ms):
		b = min(len(data), a + ms)
		# if i == n:
		# 	my_spn.model.normalize_weights()
		# if i < n:
		my_spn.model.apply_count(data[a:b])
		# elif i < 27:
		# 	my_spn.model.session.run(my_spn.model.opt_val, feed_dict = {my_spn.model.input: data[a:b]})
		# else:
		# 	my_spn.model.session.run(my_spn.model.opt_val2, feed_dict = {my_spn.model.input: data[a:b]})
		a += ms
		if ms < minibatch_cap:
			ms *= 1.6
			ms = int(ms)
		print i
		if a > len(data):
			break;
else:
	data = my_spn.data.train

	a = 0
	ms = 2
	n = 17
	for i in range(len(data)//ms):
		b = min(len(data), a + ms)
		n_data = data[a:b]
		n_data = n_data[:, my_spn.input_order, :]
		n_data = np.reshape(n_data, (len(n_data), n_data.shape[1]*n_data.shape[2]))
		# if i == n:
		# 	my_spn.model.normalize_weights()
		# if i < n:
		my_spn.model.apply_count(n_data)
		# elif i < 27:
		# 	my_spn.model.session.run(my_spn.model.opt_val, feed_dict = {my_spn.model.input: data[a:b]})
		# else:
		# 	my_spn.model.session.run(my_spn.model.opt_val2, feed_dict = {my_spn.model.input: data[a:b]})
		a += ms
		if ms < minibatch_cap:
			ms *= 1.6
			ms = int(ms)
		print i
		if a > len(data):
			break;

test_loss = my_spn.model.session.run(my_spn.model.loss, feed_dict={my_spn.model.input: my_spn.data.test.T})
valid_loss = my_spn.model.session.run(my_spn.model.loss, feed_dict={my_spn.model.input: my_spn.data.valid.T})
print "Test Loss:"
print test_loss
print "Valid Loss:"
print valid_loss

myfile = open('results/' + mn + '.py', 'w')
myfile.write("losses = {'test_loss':" + str(test_loss) + ", 'val_loss':" + str(valid_loss) + ", 'name': '"+ mn +"' }")
#for d in my_spn.data.valid.T[:100]:
	
#/float(len(my_spn.data.valid.T[:100]))
# my_spn2 = SPN()
# my_spn2.make_model_from_file('../spn_models/ + mn + .spn.txt', False)
# my_spn2.start_session()
# # print map(lambda x: x.get_shape(), my_spn2.model.weights)
# # print map(lambda x: len(x), my_spn2.model.node_layers)
# # print my_spn2.predict([[1]*len(my_spn2.input_order)*2])
# #my_spn2.make_model_from_file('../spn_models/ + mn + .spncc.spn.txt', True)
# # assert 1.1 >= my_spn2.predict([[1]]*1574)[0][0]
# my_spn2.add_data('../Data/ + mn + .ts.data')
# my_spn2.add_data('../Data/ + mn + .test.data', 'test')
# my_spn2.add_data('../Data/ + mn + .valid.data', 'valid')
# a = my_spn2.test(my_spn2.data.valid)
# print map(lambda x: np.sum(x), a)
# print map(lambda x: np.sum(x), b)
# print 'tests passed!'
