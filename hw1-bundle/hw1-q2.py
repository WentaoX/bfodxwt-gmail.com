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
    return nodes_index, features

def calculate_top_near_nodes(featuers, center_node, top_n):
    '''
    :param featuers: numpy array, each row is vector for each node
    :param center_node: the node we want to compare with each other of the rest
    :param top_n: return the number of most similar nodes
    :return: a list of index
    '''
    similarity_score = np.dot(features, featuers[center_node, :])
    # normalize
    norm_center_node = np.linalg.norm(featuers[center_node, :])
    norm_all_nodes = np.linalg.norm(featuers, axis=1)
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

nodes_index, features = create_basic_features()
nodes_id_top = calculate_top_near_nodes(features, 9, 5)
print("feature for node 9 is:")
print(features[9, :])
print('top 5 similar nodes are:')
print(nodes_id_top)
print('top 5 nodes of vectors are:')
print(features[nodes_id_top, :])