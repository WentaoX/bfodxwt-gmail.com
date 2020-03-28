import snap
import numpy as np

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

def calculate_top_near_nodes(features, center_node, top_n):
    '''
    :param featuers: numpy array, each row is vector for each node
    :param center_node: the node we want to compare with each other of the rest
    :param top_n: return the number of most similar nodes
    :return: a list of index
    '''
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


features_new = recursieve_features(Graph, features, 2)
q2(features_new)
