from model import Model 
from SPN2 import SPN


my_spn = SPN()
my_spn.make_model_from_file('../spn_models/pumsb_star.spncc.spn.txt', False)
my_spn.start_session()
print my_spn.predict([[1]]*len(my_spn.input_order)*2)
#my_spn.make_model_from_file('../spn_models/nltcs.spncc.spn.txt', True)
# assert 1.1 >= my_spn.predict([[1]]*1574)[0][0]
my_spn.add_data('../Data/pumsb_star.ts.data')
my_spn.add_data('../Data/pumsb_star.test.data', 'test')
my_spn.add_data('../Data/pumsb_star.valid.data', 'valid')
my_spn.train(20, minibatch_size=250, valid=False, test=False)
print my_spn.evaluate(my_spn.data.test, "lol")

print 'tests passed!'
