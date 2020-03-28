import snap
import numpy as np
import matplotlib.pyplot as plt


def import_data(data_path = './hw1-q2.graph'):
    G = snap.TUNGraph.Load(snap.TFIn(data_path))
    return G


def gen_egonet_edges(Graph, Node):
    Nodes = snap.TIntV()
    Nodes.Add(Node.GetId())
    # id of Node's all neighbors
    for Id in Node.GetOutEdges():
        Nodes.Add(Id)
    # Nodes now have all nodes id of egonet of Node
    results = snap.GetEdgesInOut(Graph, Nodes)
    return results


def create_basic_features(data_path = './hw1-q2.graph'):
    Graph = import_data(data_path)
    node_n = Graph.GetNodes()
    features = np.full(shape=[node_n, 3], fill_value=0)
    # keep the order of Node,
    nodes_index = []
    degrees = []
    egonet_in = []
    egonet_out = []

    # Node is iterated in the order of 0, 1, 2, ....
    for Node in Graph.Nodes():
        nodes_index.append(Node.GetId())

        # calculate degrees
        degrees.append(Node.GetDeg())

        # calculate egonet in and out edges
        results = gen_egonet_edges(Graph, Node)
        egonet_in.append(results[0])
        egonet_out.append(results[1])

    features[:, 0] = degrees
    features[:, 1] = egonet_in
    features[:, 2] = egonet_out
    return Graph, features


def calculate_similarity(features, center_node):
    similarity_score = np.dot(features, features[center_node, :])
    # normalize
    norm_center_node = np.linalg.norm(features[center_node, :])
    norm_all_nodes = np.linalg.norm(features, axis=1)
    normalizer = np.dot(norm_all_nodes, norm_center_node)

    # normalize
    similarity_score = np.divide(similarity_score, normalizer)
    # replace nan with 0
    np.nan_to_num(similarity_score, copy=False)
    assert (similarity_score[center_node] == 1)  # self similarity should be 1
    return similarity_score


def calculate_top_near_nodes(features, center_node, top_n):
    '''
    :param featuers: numpy array, each row is vector for each node
    :param center_node: the node we want to compare with each other of the rest
    :param top_n: return the number of most similar nodes
    :return: a list of index
    '''
    similarity_score = calculate_similarity(features, center_node)
    # the argsort() return sorted (ascending accroding to value) index,
    # thus select the ones in the latest part. But the self score should
    # be the largest, thus remove the self score.
    sorted_n = similarity_score.argsort()[-(top_n + 1): -1][::-1]
    print('the score for top nodes are:')
    print(similarity_score[sorted_n])
    return sorted_n


def q2(features_xy, center_node=9, top_n=5):
    nodes_id_top = calculate_top_near_nodes(features_xy, center_node, top_n)
    print("feature for node 9 is:")
    print(features_xy[center_node, :])
    print('top 5 similar nodes are:')
    print(nodes_id_top)
    print('top 5 nodes of vectors are:')
    print(features_xy[nodes_id_top, :])


# run q2.1
Graph, features = create_basic_features()
q2(features)


def recursieve_features(Graph, features, recursive_level):
    '''
    :param Graph: Graph
    :param features:  basic featurs of Graph
    :param recursive_level: int, how many steps to recurse
    :return: features with recursive features added
    '''
    while recursive_level > 0:
        features_shape = features.shape
        # the vector x2 because of mean and sum
        features_new = np.zeros(shape=[features_shape[0], features_shape[1] * 2])
        for Node in Graph.Nodes():
            current_node_id = Node.GetId()
            node_neighbors = []
            for id in Node.GetOutEdges():
                node_neighbors.append(id)
                neighbors_n = len(node_neighbors)
            if neighbors_n > 0:
                vector_neighbors = features[node_neighbors, :]

                vector_sum = np.sum(vector_neighbors, axis=0)
                vector_mean = vector_sum / neighbors_n
                vector_new = np.concatenate((vector_mean, vector_sum), axis=0)
                features_new[current_node_id, :] = vector_new
        features = np.concatenate((features, features_new), axis=1)

        recursive_level -= 1
    print (features.shape)
    return features


# run q2.2
features_new = recursieve_features(Graph, features, 2)
q2(features_new)


def find_value_node(arr, num):
    for node_id, score in enumerate(arr):
        if round(score, 2) == num:
            return node_id


def role_discovery(features, center_node):
    similarity_score = calculate_similarity(features, center_node)

    # q3_1

    plt.hist(similarity_score, bins=100)
    plt.xlabel(f'Similarity with Node {center_node}')
    plt.ylabel('Count')
    plt.title('Similarity Score Distribution')
    plt.legend()
    plt.show()

    # q3_2, select a 3 vector with similar score [000, 0.600, 0.85] seperately.

    node_dict = {}
    node_dict['dot0'] = find_value_node(similarity_score, 0)
    node_dict['dot6'] = find_value_node(similarity_score, 0.6)
    node_dict['dot85'] = find_value_node(similarity_score, 0.85)

    print(node_dict)
    for k in node_dict.keys():
        print(f'Vecotor of Node {k} is:')
        print(features[node_dict[k], :])
        print('\n')
    '''
    {'dot0': 19, 'dot6': 42, 'dot85': 256}
    Vecotor of Node dot0 is: # only publish paper himself
    [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0.]

    Vecotor of Node dot6 is: # only two author, both the two authors do not
    cooperate with others
    [1. 1. 0. 1. 1. 0. 1. 1. 0. 1. 1. 0. 1. 1. 0. 1. 1. 0. 1. 1. 0. 1. 1. 0.
     1. 1. 0.]

    Vecotor of Node dot85 is: # node have two other neigbours, two other neibours
    cooperate with each other too, and two neigbours authour cooperate with other
    authours too.
    [2.          3.          2.          3.          5.         10.
     6.         10.         20.          3.          5.         10.
     5.66666667 14.33333333 14.         17.         43.         42.
     6.         10.         20.         11.33333333 28.66666667 28.
     34.         86.         84.]
     
     '''
    
role_discovery(features_new, 9)