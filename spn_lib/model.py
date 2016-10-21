#Model file
from load_file import *
from nodes import *
import random
import tensorflow as tf
import numpy as np

class Model:

    def __init__(self, filename, random_weights=False, optimizer=tf.train.AdamOptimizer, file_weights=True):
        #filename: file to load the model from
        #optimizer: the SGD optimizer prefered
        #file_weights: A boolean representing whether to use weights from file or made up weights
        #currently doesn't allow you to build custom SPNs

        #Tensorflow Graph Variables

        self.input = None #input tensor
        self.output = None #output tensor
        self.sparse_tensors = [] #all sparse tensors to be used in graph
        self.variables = [] #all variables
        self.layer_computations = [] #all mid computations in graph
        self.norm_value = 1.0

        #Nodes loaded from files
    
        self.pos_dict, self.id_node_dict, self.node_layers, self.leaf_id_order, self.input_layers, self.input_order = load_file(filename, random_weights)

        #For training and inference

        self.loss = None #loss function, None if uninitialized
        self.optimizer = optimizer
        self.opt_val = None
        self.session = None
        self.initalizer = tf.initialize_all_variables

        #Do Work
        self.build_variables()
        self.build_forward_graph()
        self.start_session()
        self.get_normal_value()
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
        return tf.Variable(ws, dtype=tf.float64), tf.constant(s, dtype=tf.int64), tf.constant(inds, dtype=tf.int64)


    def build_variables(self):
        #Build Variables
        #Input Placeholder
        self.input = tf.placeholder(dtype=tf.float64, 
                                     shape=(len(self.input_order)*2, 
                                     None))
        #Input matrix
        input_weights, input_shape, input_indices = self.build_input_matrix(self.leaf_id_order)
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
            else:
            	print 'yo'
            	trainable = False
            shape = [len(node_layer)-self.input_layers[L], len(self.node_layers[L-1])]
            W = tf.Variable(weights, trainable=trainable, dtype=tf.float64)
            I = tf.constant(indices, dtype=tf.int64)
            S = tf.constant(shape, dtype=tf.int64)
            # print shape
            L += 1
            matrix = tf.SparseTensor(I, tf.nn.relu(tf.identity(W)), S)
            variables.append((W, I, S))
            layer_matrices.append(matrix)
        self.sparse_tensors = [input_matrix] + layer_matrices
        self.variables = variables

            
    def build_forward_graph(self):
    	computations = []

    	#the input to be appended to each layer
    	input_splits = []
    	#compute the input
    	sums_of_sparses = [tf.reshape(tf.sparse_reduce_sum(sm, reduction_axes=1), [-1 ,1]) for sm in self.sparse_tensors]
    	print sums_of_sparses
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
    			current_computation = tf.concat(0,
    								  [tf.div(tf.sparse_tensor_dense_matmul(matrix, current_computation), sums_of_sparses[i+1]), 
    								  											input_splits[L]])
    		else:
    			current_computation = tf.exp(tf.sparse_tensor_dense_matmul(matrix, tf.log(current_computation)))
    		L += 1;
    		computations.append(current_computation)

    	self.output = current_computation
    	self.loss = -tf.reduce_mean(tf.log(self.output))
    	self.opt_val = self.optimizer().minimize(self.loss)


    def get_normal_value(self):
    	ones = [[1]]*len(self.input_order)*2
    	norm_value = self.output.eval(session=self.session, feed_dict = {self.input: ones})
    	self.norm_value = norm_value
    	return norm_value


