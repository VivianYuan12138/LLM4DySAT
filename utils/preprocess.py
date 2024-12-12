
import numpy as np
# import dill
import pickle as pkl
import networkx as nx
import scipy.sparse as sp

from sklearn.model_selection import train_test_split


np.random.seed(123)

def load_graphs(dataset_str):
    """Load graph snapshots given the name of dataset"""
    with open("dataset/{}/sp_npz/{}".format(dataset_str, "graph.pkl"), "rb") as f:
        graphs = pkl.load(f) # r'D:\Pycharm\code0\model-set\DySAT_MY\DySAT_MY\DySAT_MY\dataset\MT
    print("Loaded {} graphs ".format(len(graphs)))
    # adjs = [nx.adjacency_matrix(g) for g in graphs]
    return graphs #, adjs


def get_evaluation_data(graphs):
    """ Load train/val/test examples to evaluate link prediction performance"""
    # 加载训练/验证/测试示例以评估链路预测性能
    eval_idx = len(graphs) - 2
    # eval_idx = len(graphs) - 1
    eval_graph = graphs[eval_idx]
    next_graph = graphs[eval_idx]
    print("Generating eval data ....")
    train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = \
        create_data_splits(eval_graph, next_graph, val_mask_fraction=0.2, test_mask_fraction=0.2)

    return train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false

def create_data_splits(graph, next_graph, val_mask_fraction=0.2, test_mask_fraction=0.2):
    edges_next = np.array(list(nx.Graph(next_graph).edges()))
    edges_positive = []   # Constraint to restrict new links to existing nodes.
    for e in edges_next:
        if graph.has_node(e[0]) and graph.has_node(e[1]):
            edges_positive.append(e)
    edges_positive = np.array(edges_positive) # [E, 2]
    edges_negative = negative_sample(edges_positive, graph.number_of_nodes(), next_graph)
    

    train_edges_pos, test_pos, train_edges_neg, test_neg = train_test_split(edges_positive, 
            edges_negative, test_size=val_mask_fraction+test_mask_fraction)
    val_edges_pos, test_edges_pos, val_edges_neg, test_edges_neg = train_test_split(test_pos, 
            test_neg, test_size=test_mask_fraction/(test_mask_fraction+val_mask_fraction))

    return train_edges_pos, train_edges_neg, val_edges_pos, val_edges_neg, test_edges_pos, test_edges_neg
            
def negative_sample(edges_pos, nodes_num, next_graph):
    edges_neg = []
    while len(edges_neg) < len(edges_pos):
        idx_i = np.random.randint(0, nodes_num)
        idx_j = np.random.randint(0, nodes_num)
        if idx_i == idx_j:
            continue
        if next_graph.has_edge(idx_i, idx_j) or next_graph.has_edge(idx_j, idx_i):
            continue
        if edges_neg:
            if [idx_i, idx_j] in edges_neg or [idx_j, idx_i] in edges_neg:
                continue
        edges_neg.append([idx_i, idx_j])
    return edges_neg