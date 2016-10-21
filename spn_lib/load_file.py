#loading a graph from a file

from nodes import *
import random

def split_up_file(fname):
    infile = open(fname, 'r')
    t = 'placeholder'
    lines = []
    while t != '':
        t = infile.readline()
        lines.append(t[:-1])
    n = 0
    for i in range(len(lines)):
        if "EDGES" in lines[i]:
            n = i;
            break;

    nodes = lines[1:n]
    edges = lines[n+1:]
    return nodes, edges

def build_nodes(nodes):
    big_dict = {}
    Leaves = []
    Prods = []
    Sums = []
    for l in nodes:
        if 'PRD' in l:
            arr = l.split(',')
            node = PrdNode(arr[0])
            big_dict[arr[0]] = node
            Prods.append(arr[0])
        elif 'SUM' in l:
            arr = l.split(',')
            node = SumNode(arr[0])
            big_dict[arr[0]] = node
            Sums.append(arr[0])
        elif 'LEAVE' in l:
            arr = l.split(',')
            node = Leaf(arr[0], arr[3], arr[4], arr[2])
            big_dict[arr[0]] = node
            Leaves.append(arr[0])
    return Leaves, Prods, Sums, big_dict


def add_connections(id_node_dict, edges, random_weights=False):
    for edge in edges:
        a = edge.split(',')
        if a[0] == '' or a[1] == '':
            continue
        id_node_dict[a[0]].children.append(a[1])
        id_node_dict[a[1]].parents.append(a[0])
        if len(a) == 3:
            if random_weights:
                id_node_dict[a[0]].weights.append(random.random() + 0.5)
            else:
                id_node_dict[a[0]].weights.append(a[2])
    return id_node_dict

def add_ranks(id_node_dict, leaf_id):
    currs = set(leaf_id)
    rank = 1
    while len(currs) > 0:
        prev_currs = currs
        new_currs = set()
        for s in list(currs):
            for p in id_node_dict[s].parents:
                new_currs.add(p)
            id_node_dict[s].rank = rank
        currs = new_currs
        rank += 1
    orank = rank
    rank -= 1
    currs = prev_currs
    while len(currs) > 0:
        new_currs = set()
        for s in list(currs):
            for p in id_node_dict[s].children:
                new_currs.add(p)
            id_node_dict[s].TRank = rank
        currs = new_currs
        rank -= 1
    return orank, id_node_dict

def create_layers(id_node_dict, rank):
    node_list = [[] for x in range(rank)]
    for k in id_node_dict.keys():
        n = id_node_dict[k]
        node_list[n.TRank].append(n)
    return node_list[1:]

def make_pos_dict(node_layers):
    new_dict = {}
    for i in range(len(node_layers)):
        node_layers[i].sort(lambda x, y: 1 if isinstance(x, Leaf) else -1)
        for j in range(len(node_layers[i])):
            new_dict[node_layers[i][j].id] = (i, j)
    return new_dict, node_layers

def clean_up_inputs(node_layers):
    input_layers = []
    input_order = []
    leaf_order = []
    for n_lst in node_layers:
        c = 0
        for n in n_lst:
            if isinstance(n, Leaf):
                c += 1
                input_order.append(int(n.inp))
                leaf_order.append(n.id)
        input_layers.append(c)
    return leaf_order, input_layers, input_order

def load_file(fname, random_weights=False):
    #get the node and edge strings from a file
    file_nodes, file_edges = split_up_file(fname)
    #get all the different nodes and a dict that matches id to node
    leaf_ids, prod_ids, sum_ids, id_node_dict = build_nodes(file_nodes)
    #add all the edges to the nodes
    id_node_dict = add_connections(id_node_dict, file_edges, random_weights)
    if random_weights:
        for id in sum_ids:
            summ = sum(id_node_dict[id].weights)
            id_node_dict[id].weights = map(lambda x: x/summ, id_node_dict[id].weights)
    #determine all the ranks for each node
    rank, id_node_dict = add_ranks(id_node_dict, leaf_ids)
    #turn them all into layers
    node_layers = create_layers(id_node_dict, rank)
    # print map(lambda x: len(x), node_layers)
    #create a dict for the position of every node given the ids
    pos_dict, node_layers = make_pos_dict(node_layers)
    #getting the ordering right
    leaf_id_order, input_layers, input_orders = clean_up_inputs(node_layers)

    return pos_dict, id_node_dict, node_layers, leaf_id_order, input_layers, input_orders
