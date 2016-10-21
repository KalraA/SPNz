import numpy as np

class Data:
    def __init__(self, input_order):
        self.input_order = input_order
        self.train = None
        self.valid = None
        self.test = None

    def load_data_from_file(self, fname):
        D = open(fname, 'r')
        Data = []
        i = 'lol'
        while (i != ''):
            i = D.readline()
            if i == '':
                break;
            Data.append((map(lambda x: [int(x), 0] if int(x) == 1 else [int(x), 1], i.split(','))))
        return np.array(Data)

    def load_and_process_train_data(self, fname):
        D = self.load_data_from_file(fname)
        P1 = D[:, self.input_order, :]
        P2 = np.reshape(P1, (len(P1), P1.shape[1]*P1.shape[2]))
        self.train = P2.T

    def load_and_process_valid_data(self, fname):
        D = self.load_data_from_file(fname)
        P1 = D[:, self.input_order, :]
        P2 = np.reshape(P1, (len(P1), P1.shape[1]*P1.shape[2]))
        self.valid = P2.T

    def load_and_process_test_data(self, fname):
        D = self.load_data_from_file(fname)
        P1 = D[:, self.input_order, :]
        P2 = np.reshape(P1, (len(P1), P1.shape[1]*P1.shape[2]))
        self.test = P2.T
        
