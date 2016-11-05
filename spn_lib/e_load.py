from load_file import *
import copy


def e_add_ranks(id_node_dict, leaf_id):
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
    top = None;
    currs = prev_currs
    if len(currs) == 1:
        top = list(currs);
    while len(currs) > 0:
        new_currs = set()
        for s in list(currs):
            for p in id_node_dict[s].children:
                new_currs.add(p)
            id_node_dict[s].TRank = rank
        currs = new_currs
        rank -= 1
    return top, orank, id_node_dict


def e_create_layers(id_node_dict, rank, curr):
    node_list = [[] for x in range(rank)]
    for k in id_node_dict.keys():
        n = id_node_dict[k]
        # print n.TRank
        node_list[n.TRank].append(n)
    node_proper = [[] for x in range(rank)]
    inds_proper = [[] for x in range(rank)]
    inds = [0]
    curr = [id_node_dict[curr[0]]]
    while(curr and inds):
        next_curr = []
        next_inds = []
        i = 0;
        for n, ind in zip(curr, inds):
            node_proper[n.TRank].append(n)
            inds_proper[n.TRank].append(ind)
            for c in n.children:
                next_curr.append(id_node_dict[c])
                next_inds.append(i)
            if (n.children):
            	i += 1;
        curr = next_curr;
        inds = next_inds;

    return node_proper[1:], inds_proper[1:]

def e_finish_layers(layers):
	id_map = lambda z: map(lambda y: y.id, z)
	new_layers = map(lambda lst: sorted(copy.deepcopy(lst), (lambda x, y: -1 if isinstance(x, SumNode) and isinstance(y, Leaf) else 1)), layers)
	layer_ids = map(lambda x: id_map(x), new_layers);
	# print reduce(lambda r, f: r+f.children, new_layers[4], [])
	switches = map(lambda x: map(lambda y: layer_ids[x[0]].index(y.id), x[1]), zip(range(len(layers)), layers))
	return new_layers, switches

def e_make_pos_dict(node_layers):
    new_dict = {}
    for i in range(len(node_layers)):
        for j in range(len(node_layers[i])):
            new_dict[node_layers[i][j].id] = (i, j)
    return new_dict, node_layers

def e_load(fname, random_weights=False):
    #get the node and edge strings from a file
    file_nodes, file_edges = split_up_file(fname)
    #get all the different nodes and a dict that matches id to node
    leaf_ids, prod_ids, sum_ids, id_node_dict = build_nodes(file_nodes, random_weights)
    #add all the edges to the nodes
    id_node_dict = add_connections(id_node_dict, file_edges, random_weights)
    if random_weights:
        for id in sum_ids:
            summ = sum(id_node_dict[id].weights)
            id_node_dict[id].weights = map(lambda x: x/summ, id_node_dict[id].weights)
    #determine all the ranks for each node
    
    top ,rank, id_node_dict = e_add_ranks(id_node_dict, leaf_ids)
    #turn them all into layers
    shuffle_layers, inds = e_create_layers(id_node_dict, rank, top)
    node_layers, shuffle_layers = e_finish_layers(shuffle_layers)
    # print map(lambda x: len(x), node_layers)
    #create a dict for the position of every node given the ids
    pos_dict, node_layers = e_make_pos_dict(node_layers)
    #getting the ordering right
    print "print ur mom"
    leaf_id_order, input_layers, input_orders = clean_up_inputs(node_layers)

    return pos_dict, id_node_dict, node_layers, leaf_id_order, input_layers, input_orders, shuffle_layers, inds
