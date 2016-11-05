#Model file
from load_file import *
from nodes import *
import random
import tensorflow as tf
import numpy as np
from e_load import *

# def findMaxesTensor(T, S):


class Model:

    def __init__(self, optimizer=tf.train.AdamOptimizer, file_weights=True):
        #filename: file to load the model from
        #optimizer: the SGD optimizer prefered
        #file_weights: A boolean representing whether to use weights from file or made up weights
        #currently doesn't allow you to build custom SPNs

        #Tensorflow Graph Variables
        self.summ = tf.placeholder(dtype=tf.string, shape=());
        self.loss_summary = None;
        self.writer = None;
        self.input = None #input tensor
        self.output = None #output tensor
        self.sparse_tensors = [] #all sparse tensors to be used in graph
        self.variables = [] #all variables
        self.layer_computations = [] #all mid computations in graph
        self.norm_value = 1.0
        self.counter_tensors = []
        self.weights = None
        self.counting = []
        self.updates = []

        #Nodes loaded from files
    
        self.pos_dict, self.id_node_dict, self.node_layers, self.leaf_id_order, self.input_layers, self.input_order = None, None, None, None, None, None

        #For training and inference

        self.loss = None #loss function, None if uninitialized
        self.optimizer = optimizer
        self.optimizer2 = tf.train.AdamOptimizer
        self.opt_val = None
        self.session = None
        self.initalizer = tf.initialize_all_variables
        self.shuffle = None;
        self.reverse_shuffle = None;
        self.inds = None;

        #Do Work

    def build_model_from_file(self, fname, random_weights):
        self.pos_dict, self.id_node_dict, self.node_layers, self.leaf_id_order, self.input_layers, self.input_order = load_file(fname, random_weights)

    def build_random_model(self, bfactor, input_length):
        self.pos_dict, self.id_node_dict, self.node_layers, self.leaf_id_order, self.input_layers, self.input_order = build_random_net(bfactor, input_length)

    def build_fast_model(self, bfactor, input_length):
        self.pos_dict, self.id_node_dict, self.node_layers, self.leaf_id_order, self.input_layers, self.input_order, self.shuffle, self.inds = e_load(bfactor, input_length)

    def compile(self):
        self.build_variables()
        self.build_forward_graph()
        self.start_session()
        self.writer = tf.train.SummaryWriter('logs/minibatch_range3', self.session.graph_def)
        self.get_normal_value()
        self.close_session()

    def build_input_vector(self, leafs):
        ws = []
        inds = []
        for i in range(len(leafs)):
            node = self.id_node_dict[str(leafs[i])]
            a = float(node.weights[0])
            b = float(node.weights[1])
            ws.append(a)
            ws.append(b)
            inds.append(i)
            inds.append(i)
        # print inds
        return tf.Variable(ws, dtype=tf.float64), inds

    def build_reverse_shuffle(self, shuffle):
        reverse_shuffle = []
        for s in shuffle:
            new_s = [0]*len(s)
            for i in range(len(s)):
                new_s[s[i]] = i
            reverse_shuffle.append(new_s)
        return reverse_shuffle

    def build_fast_variables(self):
        weights = []
        input_weights, input_inds = self.build_input_vector(self.leaf_id_order)
        weights.append(input_weights)
        for L in range(1, len(self.node_layers)):
            ws = []
            if isinstance(self.node_layers[L][0], SumNode):
                i = 0;
                j = 0;
                print i
                print self.node_layers[L]
                while i < len(self.node_layers[L]) and isinstance(self.node_layers[L][i], SumNode):
                    ws += map(lambda x: float(x), self.node_layers[L][i].weights);

                    print self.inds[L-1][j:j+len(self.node_layers[L][i].weights)]
                    j += len(self.node_layers[L][i].weights)
                    i += 1;
            print ws
            weights.append(tf.Variable(ws, dtype=tf.float64))
        self.weights = weights;
        self.reverse_shuffle = self.build_reverse_shuffle(self.shuffle)
        self.inds = map(lambda x: tf.constant(x), [input_inds]+self.inds)
        self.shuffle = map(lambda x: tf.constant(x), self.shuffle)
        self.reverse_shuffle = map(lambda x: tf.constant(x), self.reverse_shuffle)
        # self.input_layers = map(lambda x: [0] + tf.constant(x), self.input_layers)

    def build_fast_forward_pass(self):
        computations = []
        self.input = tf.placeholder(dtype=tf.float64, 
                                         shape=(None, len(self.input_order)*2), name='Input')
        # self.input = tf.placeholder(dtype=tf.float64, 
        #                                  shape=(len(self.input_order)*2), name='Input')
        #the input to be appended to each layer
        input_splits = []
        #compute the input
        weights = []
        for L in range(len(self.weights)):
            n = tf.constant(0.000001, dtype=tf.float64)
            weights.append(tf.add(tf.nn.relu(tf.sub(self.weights[L], n)), n))

        with tf.name_scope('NORM_FACTOR'):
            # print (map(lambda x: x.get_shape(), self.inds))
            sum_of_weights = [tf.segment_sum(x, y) if x.get_shape()[0] > 0 else None for x, y in zip(self.weights, self.inds)]
        # print sums_of_sparses
        
        with tf.name_scope('LEAFS_' + str(len(self.input_order))):
            self.counting.append(self.input)
            input_computation_w = tf.mul(self.input, weights[0])
            input_computation_s = tf.transpose(tf.segment_sum(tf.transpose(input_computation_w), self.inds[0]))
            input_computation_n = tf.div(input_computation_s, sum_of_weights[0])
            computations.append(input_computation_n)
            
        #split the input computation and figure out which one goes in each layer
            j = 0
            for i in range(len(self.input_layers)):
                a = tf.constant(j)
                b = self.input_layers[i]
                input_splits.append(tf.slice(input_computation_n, [0, a], [-1, b]))
                # input_splits.append(tf.slice(input_computation_n, [a], [b]))
                j += b;
                # print "size"
                # print size

        current_computation = input_splits[0]
# [TensorShape([Dimension(1574)]), TensorShape([Dimension(0)]), TensorShape([Dimension(28)]), TensorShape([Dimension(0)]), TensorShape([Dimension(98)]), TensorShape([Dimension(0)]), TensorShape([Dimension(16)])]
# [66, 28, 560, 98, 186, 16, 1]
# tests passed!
        print "print shit"
        for i in range(len(self.node_layers[1:])):
            L = i+1 #the layer number
            # print current_computation.get_shape()
            # print self.shuffle[i]
            # print self.inds[L].get_shape()
            print "yoo"
            
            if self.weights[L].get_shape()[0] == 0: #product
                current_computation = tf.log(current_computation)
                current_computation = tf.transpose(tf.segment_sum(tf.transpose(current_computation), self.inds[L]))
                current_computation = tf.exp(current_computation)
            else:
                self.counting.append(current_computation)
                current_computation = tf.mul(current_computation, weights[L])
                current_computation = tf.transpose(tf.segment_sum(tf.transpose(current_computation), self.inds[L]))
                current_computation = tf.div(current_computation, sum_of_weights[L])
                current_computation = tf.concat(1, [current_computation, input_splits[L]])
                current_computation = tf.transpose(tf.gather(tf.transpose(current_computation), self.shuffle[L]))
            computations.append(current_computation)

        self.output = current_computation
        with tf.name_scope('loss'):
            self.loss = -tf.reduce_mean(tf.log(self.output))
            self.loss_summary = tf.scalar_summary(self.summ, self.loss)
        self.opt_val = self.optimizer().minimize(self.loss)
        self.opt_val2 = self.optimizer2().minimize(self.loss)
        self.computations = computations

    def countss(self):
        maxed_out = []
        val = tf.constant(1.99, dtype=tf.float64)
        t = lambda x: tf.transpose(x)
        for c in range(len(self.counting)):
            maxes = tf.mul(tf.transpose(tf.segment_max(tf.transpose(self.counting[c]), self.inds[c*2])), val)
            back_maxes = tf.transpose(tf.gather(tf.transpose(maxes), self.inds[c*2]))
            counts = tf.round(tf.div(self.counting[c], back_maxes))
            maxed_out.append(counts)
        updates = []
        splits = []
        curr = tf.constant([[1.0]], dtype=tf.float64)
        for i in reversed(range(len(self.node_layers[1:]))):
            L = i+1
            if self.weights[L].get_shape()[0] == 0: #product node
                curr = tf.transpose(tf.gather(tf.transpose(curr), self.inds[L], name="mysumgather"))
                print "prd"
            else: #sum node
                print "sum"
                curr = tf.transpose(tf.gather(tf.transpose(curr), self.reverse_shuffle[L]))
                if (i != len(self.node_layers) - 2):
                    curr, split = curr[:, :-self.input_layers[L]], curr[:, -self.input_layers[L]:]
                    splits = [split] + splits;
                curr = tf.transpose(tf.gather(tf.transpose(curr), self.inds[L], name="mysumgather"))
                curr = tf.mul(curr, maxed_out[L//2])
                updates.append(tf.reduce_sum(curr, reduction_indices=0))
                

        print splits
        print curr
        print updates
        inputs = tf.concat(1, [curr] + splits);
        gathered = tf.transpose(tf.gather(tf.transpose(inputs), self.inds[0]))
        updates.append(tf.reduce_sum(tf.mul(gathered, self.input), reduction_indices=0))
        self.updates = updates;

    def apply_count(self, data):
        updates = self.session.run(self.updates, {self.input: data})
        # print map(lambda x: np.sum(x), updates)
        # print updates

        weights = filter(lambda x: x.get_shape()[0] > 0, self.weights)
        for i in range(len(weights)):
            u, w = list(reversed(updates))[i], weights[i]
            z = tf.assign_add(w, u)
            self.session.run(z)
        # print updates

    def fast_compile(self):
        print "print shit2"
        self.build_fast_variables()
        print "print shit"
        self.build_fast_forward_pass()
        self.countss()
        self.start_session()
        self.writer = tf.train.SummaryWriter('logs/fast/try1', self.session.graph)
        self.close_session()
    def start_session(self):
        assert self.session == None
        self.session = tf.Session()
        self.session.run(self.initalizer())
        return self.session

    def close_session(self):
        assert self.session != None

        self.session.close()
        self.session = None
        return None

    def build_input_matrix(self, leafs):
        inds = []
        ws = []
        s = []
        for i in range(len(leafs)):
            node = self.id_node_dict[str(leafs[i])]
            a = float(node.weights[0])
            b = float(node.weights[1])
            inds.append([i, i*2])
            inds.append([i, i*2+1])
            ws.append(a)
            ws.append(b)
        s = [len(leafs), len(leafs)*2]
        # print inds
        return tf.Variable(ws, dtype=tf.float64), tf.constant(s, dtype=tf.int64), tf.constant(inds, dtype=tf.int64)


    def build_variables(self):
        #Build Variables
        #Input Placeholder
        with tf.name_scope('inputs'):
            self.input = tf.placeholder(dtype=tf.float64, 
                                         shape=(len(self.input_order)*2, 
                                         None), name='Input')
            #Input matrix
            input_weights, input_shape, input_indices = self.build_input_matrix(self.leaf_id_order)
            input_counter = tf.constant([1.0]*len(self.input_order)*2, dtype=tf.float64)
            input_counter_matrix = tf.SparseTensor(input_indices, input_counter, input_shape)
            self.counter_tensors.append(input_counter_matrix)
            input_matrix = tf.SparseTensor(input_indices, tf.add(tf.nn.relu(tf.identity(input_weights)), 0.001), input_shape)

        #Layer Matrices
        layer_matrices = []
        variables = []
        L = 1
        for node_layer in self.node_layers[1:]:
            indices = []
            weights = []
            shape = []
            for i in range(len(node_layer)-self.input_layers[L]):
                for j in range(len(node_layer[i].children)):
                    #get the layer position of the child node
                    a, b = self.pos_dict[node_layer[i].children[j]]
                    indices.append([i, b])
                    if isinstance(node_layer[i], SumNode): 
                        #Sum Node
                        weights.append(float(node_layer[i].weights[j]))
                    else:
                        #Product Node
                        weights.append(1.0)
            if isinstance(node_layer[0], SumNode):
                trainable = True
                name = 'SUM_VARS' + str(L)
            else:
                trainable = False
                name = 'PROD_VARS' + str(L)
            print weights
            winds = zip(weights, indices);
            winds.sort(lambda x, y: -1 if x[1][1] < y[1][1] else 1);
            weights, indices = zip(*winds);
            print indices;
            shape = [len(node_layer)-self.input_layers[L], len(self.node_layers[L-1])]
            with tf.name_scope(name):
                W = tf.Variable(weights, trainable=trainable, dtype=tf.float64)
                I = tf.constant(indices, dtype=tf.int64)
                S = tf.constant(shape, dtype=tf.int64)
                C = tf.constant([1.0]*len(weights), dtype=tf.float64)
                # print shape
                L += 1
                matrix = tf.SparseTensor(I, tf.nn.relu(tf.identity(W)), S)
                counter_matrix = tf.SparseTensor(I, C, S);
            variables.append((W, I, S, shape))
            self.counter_tensors.append(counter_matrix)
            layer_matrices.append(matrix)
        self.sparse_tensors = [input_matrix] + layer_matrices
        self.variables = [(input_weights, input_indices, input_shape, [len(self.input_order), len(self.input_order)*2])] + variables

    def build_forward_graph(self):
        computations = []

        #the input to be appended to each layer
        input_splits = []
        #compute the input
        with tf.name_scope('NORM_FACTOR'):
            sums_of_sparses = [tf.reshape(tf.sparse_reduce_sum(sm, reduction_axes=1), [-1 ,1]) for sm in self.sparse_tensors]
        # print sums_of_sparses
        with tf.name_scope('LEAFS_' + str(len(self.input_order))):
            input_computation = tf.div(tf.sparse_tensor_dense_matmul(self.sparse_tensors[0], self.input), sums_of_sparses[0])
            computations.append(input_computation)

        #split the input computation and figure out which one goes in each layer
            i = 0
            for size in self.input_layers:
                input_splits.append(input_computation[i:i+size])
                i += size

        current_computation = input_splits[0]
        L = 1
        for i in range(len(self.node_layers[1:])):
            node_layer = self.node_layers[i+1]
            matrix = self.sparse_tensors[i+1]
            if isinstance(node_layer[0], SumNode):
                with tf.name_scope('SUM_' + str(self.variables[i+1][3][0])):
                    current_computation = tf.concat(0,
                                          [tf.div(tf.sparse_tensor_dense_matmul(matrix, current_computation, name='ComputeSum'), sums_of_sparses[i+1], name='Normalize'), 
                                                                                      input_splits[L]], name='ConcatenateInputs')
            else:
                with tf.name_scope('PROD_' + str(self.variables[i+1][3][0])):
                    current_computation = tf.exp(tf.sparse_tensor_dense_matmul(matrix, tf.log(current_computation, name='ToLogDomain'), name='ComputeProd'), name='ToNormalDomain')
            L += 1;
            computations.append(current_computation)

        self.output = current_computation
        with tf.name_scope('loss'):
            self.loss = -tf.reduce_mean(tf.log(self.output))
            self.loss_summary = tf.scalar_summary(self.summ, self.loss)
        self.opt_val = self.optimizer(0.0001).minimize(self.loss)
        self.opt_val2 = self.optimizer2(0.0001).minimize(self.loss)
        self.computations = computations
    # def counting_step(self) :


    def get_normal_value(self):
        ones = [[1]*len(self.input_order)*2]
        norm_value = self.output.eval(session=self.session, feed_dict = {self.input: ones})
        self.norm_value = norm_value
        return norm_value


