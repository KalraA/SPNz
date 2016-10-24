from model import Model 
from SPN2 import SPN


my_spn = SPN()
my_spn.make_model_from_file('../spn_models/nltcs.spn.txt', True)
my_spn.start_session()
print my_spn.predict([[1]]*len(my_spn.input_order)*2)
#my_spn.make_model_from_file('../spn_models/nltcs.spncc.spn.txt', True)
# assert 1.1 >= my_spn.predict([[1]]*1574)[0][0]
my_spn.add_data('../Data/nltcs.ts.data')
my_spn.add_data('../Data/nltcs.test.data', 'test')
my_spn.add_data('../Data/nltcs.valid.data', 'valid')
my_spn.train(25, minibatch_size=8)
print 'tests passed!'