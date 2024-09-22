import numpy as np

from causallearn.graph.Dag import Dag
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GeneralGraph import GeneralGraph


def dag2cpdag(G: Dag) -> GeneralGraph:
    """
    Covert a DAG to its corresponding PDAG

    Parameters
    ----------
    G : Direct Acyclic Graph

    Returns
    -------
    CPDAG : Completed Partially Direct Acyclic Graph

    Authors
    -------
    Yuequn Liu@dmirlab, Wei Chen@dmirlab, Kun Zhang@CMU
    """
    print("test_ 0")
    # order the edges in G
    # Get the causal ordering
    print("Get Causal Ordering-----------")
    causal_ordering = G.get_causal_ordering()
    # Initialize an empty list to store the node values
    print("test 00")
    nodes_order = []
    # Iterate over each node in the causal ordering
    for x in causal_ordering:
        # Append the corresponding value from G.node_map to the nodes_order list
        nodes_order.append(G.node_map[x])

    # nodes_order = list(
    #     map(lambda x: G.node_map[x], G.get_causal_ordering()))  # Perform a topological sort on the nodes of G
    print("test 01")
    # nodes_order(1) is the node which has the highest order
    # nodes_order(N) is the node which has the lowest order
    edges_order = np.mat([[], []], dtype=np.int64).T
    print("test 02")
    # edges_order(1,:) is the edge which has the highest order
    # edges_order(M,:) is the edge which has the lowest order
    M = G.get_num_edges()  # the number of edges in this DAG
    print("test 03")
    N = G.get_num_nodes()  # the number of nodes in this DAG
    print("test 04")
    i, j = 0, 0
    while edges_order.shape[0] < M:
        print("test 1")
        for ny in range(N - 1, -1, -1):
            j = nodes_order[ny]
            inci_all = np.where(G.graph[j, :] == 1)[0]  # all the edges that incident to j
            if len(inci_all) != 0:
                if len(edges_order) != 0:
                    inci = edges_order[np.where(edges_order[:, 1] == j)[0], 0]  # ordered edge that incident to j
                    if len(set(inci_all) - set(inci.T.tolist()[0])) != 0:
                        break
                else:
                    break
        print("test 2")
        for nx in range(N):
            i = nodes_order[nx]
            if len(edges_order) != 0:
                if (len(np.intersect1d(np.where(edges_order[:, 1] == j)[0],
                                       np.where(edges_order[:, 0] == i)[0])) == 0 and G.graph[j, i] == 1):
                    break
            else:
                if G.graph[j, i] == 1:
                    break
        print("test 3")
        edges_order = np.r_[edges_order, np.mat([i, j])]

    ## ----------------------------------------------------------------
    sign_edges = np.zeros(M)  # 0 means unknown, 1 means compelled, -1 means reversible
    while len(np.where(sign_edges == 0)[0]) != 0:
        print("test 4")
        ss = 0
        for m in range(M - 1, -1, -1):  # let x->y be the lowest ordered edge that is labeled "unknown"
            if sign_edges[m] == 0:
                i = edges_order[m, 0]
                j = edges_order[m, 1]
                break
        print("test 5")
        idk = np.where(edges_order[:, 1] == i)[0]
        k = edges_order[idk, 0]  # w->x
        for m in range(len(k)):
            if sign_edges[idk[m]] == 1:
                if G.graph[j, k[m]] != 1:  # if w is not a parent of y
                    _id = np.where(edges_order[:, 1] == j)[0]  # label every edge that incident into y with "complled"
                    sign_edges[_id] = 1
                    ss = 1
                    break
                else:
                    _id = np.intersect1d(np.where(edges_order[:, 0] == k[m, 0])[0],
                                         np.where(edges_order[:, 1] == j)[0])  # label w->y with "complled"
                    sign_edges[_id] = 1
        print("test 6")
        if ss:
            continue

        z = np.where(G.graph[j, :] == 1)[0]
        if (len(np.intersect1d(np.setdiff1d(z, i),
                               np.union1d(np.union1d(np.where(G.graph[i, :] == 0)[0], np.where(G.graph[i, :] == -1)[0]),
                                          np.intersect1d(np.where(G.graph[i, :] == -1)[0],
                                                         np.where(G.graph[:, i] == -1)[0])))) != 0):
            _id = np.intersect1d(np.where(edges_order[:, 0] == i)[0], np.where(edges_order[:, 1] == j)[0])
            sign_edges[_id] = 1  # label x->y  with "compelled"
            id1 = np.where(edges_order[:, 1] == j)[0]
            id2 = np.intersect1d(np.where(sign_edges == 0)[0], id1)
            sign_edges[id2] = 1  # label all "unknown" edges incident into y  with "complled"
        else:
            _id = np.intersect1d(np.where(edges_order[:, 0] == i)[0], np.where(edges_order[:, 1] == j)[0])
            sign_edges[_id] = -1  # label x->y with "reversible"

            id1 = np.where(edges_order[:, 1] == j)[0]
            id2 = np.intersect1d(np.where(sign_edges == 0)[0], id1)
            sign_edges[id2] = -1  # label all "unknown" edges incident into y with "reversible"

    # create CPDAG accoring the labelled edge
    nodes = G.get_nodes()
    CPDAG = GeneralGraph(nodes)
    print("test 7")
    for m in range(M):
        if sign_edges[m] == 1:
            CPDAG.add_edge(Edge(nodes[edges_order[m, 0]], nodes[edges_order[m, 1]], Endpoint.TAIL, Endpoint.ARROW))
        else:
            CPDAG.add_edge(Edge(nodes[edges_order[m, 0]], nodes[edges_order[m, 1]], Endpoint.TAIL, Endpoint.TAIL))

    print("test 8")
    return CPDAG
