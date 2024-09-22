import logging
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from copy import deepcopy

import numpy as np
import pandas as pd

from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GeneralGraph import GeneralGraph


def pdag2dag(G: GeneralGraph) -> GeneralGraph:
    # with open('variables.pkl', 'wb') as file:
    #     pickle.dump(G, file)

    with open('variables.pkl', 'rb') as file:
        G = pickle.load(file)
    #print(loaded_data)

    """
    Covert a PDAG to its corresponding DAG

    Parameters
    ----------
    G : Partially Direct Acyclic Graph

    Returns
    -------
    Gd : Direct Acyclic Graph
    """
    #logger.info("Starting PDAG to DAG conversion")

    nodes = G.get_nodes()
    # first create a DAG that contains all the directed edges in PDAG
    #print("-----Nodes-----", G)
    Gd = deepcopy(G)
    edges = Gd.get_graph_edges()
    #print("-----Edges-----")
    for edge in edges:
        #print("-----first for loop-----")
        if not ((edge.endpoint1 == Endpoint.ARROW and edge.endpoint2 == Endpoint.TAIL) or (
                edge.endpoint1 == Endpoint.TAIL and edge.endpoint2 == Endpoint.ARROW)):
            Gd.remove_edge(edge)

    Gp = deepcopy(G)
    inde = np.zeros(Gp.num_vars, dtype=np.dtype(int))  # index whether the ith node has been removed. 1:removed; 0: not
    #print('----------inde---------', inde)
    #print('Len of inde {0}', len(inde))
    inde2 = np.copy(inde)
    counter = 0
    insideCounter = 1
    while 0 in inde:
        print("-----while loop-----")
        #print('----------inde---------', inde)
        if np.array_equal(inde, inde2):
            insideCounter += 1
            # if (insideCounter == 10):
            #     inde[:] = 1
            print("Equal {0}", insideCounter)
        else:
            counter += 1
            print(counter)
        
        inde2 = np.copy(inde)
        for i in range(Gp.num_vars):
            #print("-----second for loop-----")
            if inde[i] == 0:
                print("-----if inde passed-----")
                sign = 0
                #print(Gp.graph)

                #df = pd.DataFrame(Gp.graph)
                #df = pd.DataFrame(Gp.graph, index=range(Gp.graph.shape[0]), columns=range(Gp.graph.shape[1]))
                #df.to_csv('adjacency_matrix.csv', index=True)
                #print(np.where(Gp.graph[:, i] == 1)[0])
                #input("")
                #print(np.where(inde == 0)[0])
                #input("")
                if (len(np.intersect1d(np.where(Gp.graph[:, i] == 1)[0],
                                       np.where(inde == 0)[0])) == 0):  # Xi has no out-going edges
                    #print("-----second if inside-----")
                    #input("")
                    sign = sign + 1
                    Nx = np.intersect1d(
                        np.intersect1d(np.where(Gp.graph[:, i] == -1)[0], np.where(Gp.graph[i, :] == -1)[0]),
                        np.where(inde == 0)[0])  # find the neighbors of Xi in P
                    Ax = np.intersect1d(np.union1d(np.where(Gp.graph[i, :] == 1)[0], np.where(Gp.graph[:, i] == 1)[0]),
                                        np.where(inde == 0)[0])  # find the adjacent of Xi in P
                    Ax = np.union1d(Ax, Nx)
                    if len(Nx) > 0:
                        if check2(Gp, Nx, Ax):  # according to the original paper
                            sign = sign + 1
                        else:
                            print("Not check2---------", insideCounter)
                            if insideCounter % 10 == 0:
                                sign = sign + 1
                    else:
                        sign = sign + 1
                else:
                    if insideCounter % 10 == 0:
                        sign = 2
                if sign == 2:
                    print("-----if sign = 2 passed-----")
                    # for each undirected edge Y-X in PDAG, insert a directed edge Y->X in G
                    for index in np.intersect1d(np.where(Gp.graph[:, i] == -1)[0], np.where(Gp.graph[i, :] == -1)[0]):
                        Gd.add_edge(Edge(nodes[index], nodes[i], Endpoint.TAIL, Endpoint.ARROW))
                    inde[i] = 1
                    print("-----if sign 2 for loop end-----")
    #print("-----while loop ends-----")
    return Gd


def check2(G: GeneralGraph, Nx, Ax):
    s = 1
    for i in range(len(Nx)):
        j = np.delete(Ax, np.where(Ax == Nx[i])[0])
        if len(np.where(G.graph[Nx[i], j] == 0)[0]) != 0:
            s = 0
            break
    return s

