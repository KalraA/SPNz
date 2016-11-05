from model import Model 
from SPN2 import SPN
import numpy as np
import random
my_spn = SPN()
my_spn.make_fast_model_from_file('../spn_models/nltcs.spn.txt', True)
print "print shit"
my_spn.start_session()
print map(lambda x: x.get_shape(), my_spn.model.weights)
print map(lambda x: len(x), my_spn.model.node_layers)
print my_spn.predict([[1]*len(my_spn.input_order)*2])
#my_spn.make_model_from_file('../spn_models/nltcs.spncc.spn.txt', True)
# assert 1.1 >= my_spn.predict([[1]]*1574)[0][0]
my_spn.add_data('../Data/nltcs.ts.data')
my_spn.add_data('../Data/nltcs.test.data', 'test')
my_spn.add_data('../Data/nltcs.valid.data', 'valid')
# b = my_spn.test(my_spn.data.valid.T)
# my_spn.train(20)
i = 0
# for d in my_spn.data.valid.T[:100]:
# 	i += 1
# 	my_spn.model.apply_count([d])
# 	if i % 100 == 0:
# 		print str(i) + "/" + str(len(my_spn.data.valid.T))
data = my_spn.data.train.T
a = 0
ms = 1
for i in range(len(data)//ms):
	b = min(len(data), a + ms)
	my_spn.model.apply_count(data[a:b])
	a += ms
	if a > len(data):
		break;
loss = 0

for d in my_spn.data.valid.T[:100]:
	
	a = my_spn.model.session.run(my_spn.model.loss, feed_dict={my_spn.model.input: [d]})
	loss += a
print loss/float(len(my_spn.data.valid.T[:100]))
# my_spn2 = SPN()
# my_spn2.make_model_from_file('../spn_models/nltcs.spn.txt', False)
# my_spn2.start_session()
# # print map(lambda x: x.get_shape(), my_spn2.model.weights)
# # print map(lambda x: len(x), my_spn2.model.node_layers)
# # print my_spn2.predict([[1]*len(my_spn2.input_order)*2])
# #my_spn2.make_model_from_file('../spn_models/nltcs.spncc.spn.txt', True)
# # assert 1.1 >= my_spn2.predict([[1]]*1574)[0][0]
# my_spn2.add_data('../Data/nltcs.ts.data')
# my_spn2.add_data('../Data/nltcs.test.data', 'test')
# my_spn2.add_data('../Data/nltcs.valid.data', 'valid')
# a = my_spn2.test(my_spn2.data.valid)
# print map(lambda x: np.sum(x), a)
# print map(lambda x: np.sum(x), b)
# print 'tests passed!'