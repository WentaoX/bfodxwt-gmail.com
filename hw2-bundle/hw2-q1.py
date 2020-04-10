def probabilistic_classifier(scores, adjacency_list, ground_truth_node):
    '''
    return average, because weight is 1, thus omitted weight
    :param scores:
    :param adjacency_list:
    :return:
    '''
    for node in range(1, 11):
        if node in ground_truth_node:
            continue
        neibs = adjacency_list[node]
        if len(neibs) == 0:
            continue
        avg = 0
        for neib in neibs:
            avg += scores[neib]
        avg = avg/len(neibs)
        scores[node] = avg
    return scores

def relational_classification():
    adjacency_list = {
        1: {2, 3},
        2: {1, 3, 4},
        3: {1, 2, 6},
        4: {2, 7, 8},
        5: {6, 8, 9},
        6: {3, 5, 9, 10},
        7: {4, 8},
        8: {4, 5, 7, 9},
        9: {5, 6, 8, 10},
        10: {6, 9}
    }
    scores = {i: 0.5 for i in range(1,11)}

    # initialize ground truth
    scores[3] = 1
    scores[5] = 1
    scores[8] = 0
    scores[10] = 0

    # ground truth nodes
    ground_truth_node = {3, 5, 8, 10}

    # after second iteration
    iter_n = 2
    for i in range(iter_n):
        scores = probabilistic_classifier(scores, adjacency_list, ground_truth_node)
    print('scores after second iteration:\n', scores)

    # if threshold is 0,5, the negative nodes:
    threshold = 0.5
    neg_nodes = [node for node, value in scores.items() if value < threshold]
    print('negative nodes are:\n', neg_nodes)

relational_classification()