from model import Model 
from SPN2 import SPN

my_model = Model('nltcs.spncc.spn.txt')
# assert 7 == len(my_model.node_layers)
# assert 7 == len(my_model.sparse_tensors)
print my_model.norm_value
my_spn = SPN()
my_spn.make_model_from_file('nltcs.spn.txt', True)
my_spn.start_session()
# assert 1.1 >= my_spn.predict([[1]]*1574)[0][0]
my_spn.add_data('nltcs.ts.data')
my_spn.add_data('nltcs.test.data', 'test')
my_spn.add_data('nltcs.valid.data', 'valid')
my_spn.train(50)
print my_spn.evaluate(my_spn.data.test)
print my_spn.model.norm_value


print 'tests passed!'