import math
import numpy as np
import csv


from typing import Optional
from causallearn.score.LocalScoreFunctionClass import LocalScoreClass
from causallearn.score.LocalScoreFunction import *
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.utils.DAG2CPDAG import dag2cpdag
from causallearn.utils.GESUtils import *
from causallearn.utils.PDAG2DAG import pdag2dag


from causallearn.graph.AdjacencyConfusion import AdjacencyConfusion
from causallearn.graph.ArrowConfusion import ArrowConfusion
from causallearn.graph.SHD import SHD
#from causallearn.search.ScoreBased.GES import ges
from causallearn.score.LocalScoreFunction import local_score_BIC
from causallearn.utils.DAG2CPDAG import dag2cpdag
from causallearn.utils.TXT2GeneralGraph import txt2generalgraph
from math import log


def ges(X: ndarray, score_func: str = 'local_score_BIC', maxP: Optional[float] = None,
        parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Perform greedy equivalence search (GES) algorithm

    Parameters
    ----------
    X : data set (numpy ndarray), shape (n_samples, n_features). The input data, where n_samples is the number of samples and n_features is the number of features.
    score_func : the string name of score function. (str(one of 'local_score_CV_general', 'local_score_marginal_general',
                    'local_score_CV_multi', 'local_score_marginal_multi', 'local_score_BIC', 'local_score_BDeu')).
    maxP : allowed maximum number of parents when searching the graph
    parameters : when using CV likelihood,
                  parameters['kfold']: k-fold cross validation
                  parameters['lambda']: regularization parameter
                  parameters['dlabel']: for variables with multi-dimensions,
                               indicate which dimensions belong to the i-th variable.

    Returns
    -------
    Record['G']: learned causal graph, where Record['G'].graph[j,i]=1 and Record['G'].graph[i,j]=-1 indicates  i --> j ,
                    Record['G'].graph[i,j] = Record['G'].graph[j,i] = -1 indicates i --- j.
    Record['update1']: each update (Insert operator) in the forward step
    Record['update2']: each update (Delete operator) in the backward step
    Record['G_step1']: learned graph at each step in the forward step
    Record['G_step2']: learned graph at each step in the backward step
    Record['score']: the score of the learned graph
    """

    if X.shape[0] < X.shape[1]:
        warnings.warn("The number of features is much larger than the sample size!")

    # I commented out this line for using the new score function
    #X = np.mat(X)
        
    if score_func == 'local_score_CV_general':  # % k-fold negative cross validated likelihood based on regression in RKHS
        if parameters is None:
            parameters = {'kfold': 10,  # 10 fold cross validation
                          'lambda': 0.01}  # regularization parameter
        if maxP is None:
            maxP = X.shape[1] / 2  # maximum number of parents
        N = X.shape[1]  # number of variables
        localScoreClass = LocalScoreClass(data=X, local_score_fun=local_score_cv_general, parameters=parameters)

    elif score_func == 'local_score_marginal_general':  # negative marginal likelihood based on regression in RKHS
        parameters = {}
        if maxP is None:
            maxP = X.shape[1] / 2  # maximum number of parents
        N = X.shape[1]  # number of variables
        localScoreClass = LocalScoreClass(data=X, local_score_fun=local_score_marginal_general, parameters=parameters)

    elif score_func == 'local_score_CV_multi':  # k-fold negative cross validated likelihood based on regression in RKHS
        # for data with multi-variate dimensions
        if parameters is None:
            parameters = {'kfold': 10, 'lambda': 0.01, 'dlabel': {}}  # regularization parameter
            for i in range(X.shape[1]):
                parameters['dlabel']['{}'.format(i)] = i
        if maxP is None:
            maxP = len(parameters['dlabel']) / 2
        N = len(parameters['dlabel'])
        localScoreClass = LocalScoreClass(data=X, local_score_fun=local_score_cv_multi, parameters=parameters)

    elif score_func == 'local_score_marginal_multi':  # negative marginal likelihood based on regression in RKHS
        # for data with multi-variate dimensions
        if parameters is None:
            parameters = {'dlabel': {}}
            for i in range(X.shape[1]):
                parameters['dlabel']['{}'.format(i)] = i
        if maxP is None:
            maxP = len(parameters['dlabel']) / 2
        N = len(parameters['dlabel'])
        localScoreClass = LocalScoreClass(data=X, local_score_fun=local_score_marginal_multi, parameters=parameters)

    elif score_func == 'local_score_BIC_NEW' or score_func == 'local_score_BIC' or score_func == 'local_score_BIC_from_cov':  # Greedy equivalence search with BIC score
        if maxP is None:
            maxP = X.shape[1] / 2
        N = X.shape[1]  # number of variables
        parameters = {}
        parameters["lambda_value"] = 2
        localScoreClass = LocalScoreClass(data=X, local_score_fun=local_score_BIC, parameters=parameters)

    elif score_func == 'local_score_BDeu':  # Greedy equivalence search with BDeu score
        if maxP is None:
            maxP = X.shape[1] / 2
        N = X.shape[1]  # number of variables
        localScoreClass = LocalScoreClass(data=X, local_score_fun=local_score_BDeu, parameters=None)

    else:
        raise Exception('Unknown function!')
    score_func = localScoreClass

    node_names = [("X%d" % (i + 1)) for i in range(N)]
    nodes = []

    for name in node_names:
        node = GraphNode(name)
        nodes.append(node)

    #Create a matrix of NxN and score each cell except the diagonal ones

    # Initialize an empty NxN matrix
    matrix = np.zeros((N, N))

    # Fill the matrix with the score except for the diagonal cells
    for i in range(N):
        for j in range(N):
            if i != j:  # Exclude diagonal cells
                matrix[i, j] = single_edge_score_BIC(X, i, [j])

    # Matrix_to_tuple conversion starts here
                            
    result = []
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            result.append((i, j, matrix[i][j]))      

    # Matrix_to_tuple conversion ends here    

    # Discard 0 from the tuple

    result = []
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            cell_value = matrix[i][j]
            if cell_value != 0:
                result.append((i, j, cell_value))
 

    #CAUSAL ORDER based edge set refinement

    # Assuming ordered_nodes is your ordered list of nodes
    ordered_nodes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]

    # Create a dictionary to map nodes to their positions in the ordered list
    node_position = {node: index for index, node in enumerate(ordered_nodes)}

    result = []

    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            cell_value = matrix[i][j]
            if cell_value != 0:
                # Ensure i comes before j in the ordered list
                if node_position[i] < node_position[j]:
                    result.append((i, j, cell_value))
                    #print(result)

    # result now contains edges that comply with the ordered list of nodes

    # SORT the tuple in descending order based on the cell value for Forward Search
    result_tuple_sorted = sorted(result, key=lambda x: x[2], reverse=True)

    # Print the sorted list
    c = 0
    for item in result_tuple_sorted:
        c = c + 1
        print("item:", c, ", i:", item[0], ", j:", item[1], ", i --> j:", item[2])


    G = GeneralGraph(nodes)

    #PRIOR_KNOWLEDGE_INCORPORATION STARTS

    rows, cols = 100, 100
    pk = [[0 for x in range(rows)] for y in range(cols)]

    #1 = K edges and 2 = U edges and 3 = F edges

    #PRIOR_KNOWLEDGE_INCORPORATION ENDS
    
    # G = np.matlib.zeros((N, N)) # initialize the graph structure
    score = score_g(X, G, score_func, parameters)  # initialize the score

    G = pdag2dag(G)
    G = dag2cpdag(G)

    p = 0
    weight_lambda = -100

    
    print('----------------Sorting-based Forward Search Starts------------------')
    record_local_score = [[] for i in range(
        N)]  # record the local score calculated each time. Thus when we transition to the second phase,
    # many of the operators can be scored without an explicit call the the scoring function
    # record_local_score{trial}{j} record the local scores when Xj as a parent
    score_new = score
    count1 = 0
    counter = 0
    update1 = []
    G_step1 = []
    score_record1 = []
    graph_record1 = []


    for item in result_tuple_sorted:
        singleRowData = []
        counter = counter + 1
        count1 = count1 + 1
        score = score_new
        score_record1.append(score)
        graph_record1.append(G)
        min_chscore = 1e7
        min_desc = []
        i = item [0]
        j = item [1]
        

        # if pk[i][j] == 1: 
        #     G.graph[ i , j ] = Endpoint.TAIL.value
        #     G.graph[ j , i ] = Endpoint.ARROW.value
        #     continue
        # elif pk[i][j] == 3:
        #     G.graph[ i , j ] = Endpoint.NULL.value
        #     G.graph[ j , i ] = Endpoint.NULL.value
        #     continue
        if (G.graph[i, j] == Endpoint.NULL.value and G.graph[j, i] == Endpoint.NULL.value
                    and i != j and len(np.where(G.graph[j, :] == Endpoint.ARROW.value)[0]) <= maxP):
                # find a pair (Xi, Xj) that is not adjacent in the current graph , and restrict the number of parents
                print("----------Yes inside the if block and current i, j and counter:---------", i, j, counter)
                
                Tj = np.intersect1d(np.where(G.graph[:, j] == Endpoint.TAIL.value)[0],
                                np.where(G.graph[j, :] == Endpoint.TAIL.value)[0])  # neighbors of Xj
                Ti = np.union1d(np.where(G.graph[:, i] != Endpoint.NULL.value)[0],
                                np.where(G.graph[i, :] != Endpoint.NULL.value)[0])  # adjacent to Xi
                NTi = np.setdiff1d(np.arange(N), Ti)
                T0 = np.intersect1d(Tj, NTi)  # find the neighbours of Xj that are not adjacent to Xi
                # for any subset of T0
                sub = Combinatorial(T0.tolist())  # find all the subsets for T0
                S = np.zeros(len(sub))
                # S indicate whether we need to check sub{k}.
                # 0: check both conditions.
                # 1: only check the first condition
                # 2: check nothing and is not valid.
                

                for k in range(len(sub)):
                    if (S[k] < 2):  # S indicate whether we need to check subset(k)
                        V1 = insert_validity_test1(G, i, j, sub[k])  # Insert operator validation test:condition 1
                        if (V1):
                            if (not S[k]):
                                V2 = insert_validity_test2(G, i, j, sub[k])  # Insert operator validation test:condition 2
                                # if V2 == 0:      #For indexes like 8, 14 where k could be 85 or more
                                #     break
                            else:
                                V2 = 1

                            if (V2):
                                Idx = find_subset_include(sub[k], sub)  # find those subsets that include sub(k)
                                S[np.where(Idx == 1)] = 1
                                chscore, desc, record_local_score = insert_changed_score(X, G, i, j, sub[k],
                                                                                            record_local_score,
                                                                                            score_func,
                                                                                            parameters)
                                #print("Current i and j", i, j)
                                # calculate the changed score after Insert operator
                                # desc{count} saves the corresponding (i,j,sub{k})
                                # sub{k}:
                                if (chscore < min_chscore):
                                    min_chscore = chscore
                                    min_desc = desc
                        else:
                            Idx = find_subset_include(sub[k], sub)  # find those subsets that include sub(k)
                            S[np.where(Idx == 1)] = 2

        if (pk[i][j] == 1 and G.graph[ i , j ] != Endpoint.TAIL.value and G.graph[ j , i ] != Endpoint.ARROW.value):
            p = p + 1
            penalty = weight_lambda*log(p)
            print("----------Current Penalty is:---------", penalty)
        
        else:
            penalty = weight_lambda*1
            #print("----------Current Penalty for K-edges is:---------", penalty)

        # if (pk[i][j] == 2 and G.graph[ i , j ] != Endpoint.TAIL.value and G.graph[ j , i ] != Endpoint.TAIL.value):
        #     p = p + 1
        #     penalty = weight_lambda*log(p)
        #     print("----------Current Penalty is:---------", penalty)
        
        # else:
        #     penalty = weight_lambda*1
        #     #print("----------Current Penalty for U-edges is:---------", penalty)

        # if (pk[i][j] == 3 and G.graph[ i , j ] != Endpoint.NULL.value and G.graph[ j , i ] != Endpoint.NULL.value):
        #     p = p + 1
        #     penalty = weight_lambda*log(p)
        #     print("----------Current Penalty is:---------", penalty)
        
        # else:
        #     penalty = weight_lambda*1
        #     #print("----------Current Penalty for F-edges is:---------", penalty)

        if (len(min_desc) != 0):
            score_new = score + min_chscore + penalty
            # if (score - score_new <= 0):
            #      break
            G = insert(G, min_desc[0], min_desc[1], min_desc[2])
            update1.append([min_desc[0], min_desc[1], min_desc[2]])
            G = pdag2dag(G)
            G = dag2cpdag(G)
            G_step1.append(G)

        else:
            score_new = score #score_new = 100  #No change in performance occurs by keeping the score_new value fixed at 100
            #break
            continue     
                

    print('--------------Backward Search Starts-------------')
    count2 = 0
    score_new = score
    update2 = []
    G_step2 = []
    score_record2 = []
    graph_record2 = []

    while True:
        singleRowData = []
        count2 = count2 + 1
        score = score_new
        score_record2.append(score)
        graph_record2.append(G)
        min_chscore = 1e7
        min_desc = []
        for i in range(N):
            for j in range(N):
                # if pk[i][j] == 1: #Ignore PK --> edges
                #     G.graph[ i , j ] = Endpoint.TAIL.value
                #     G.graph[ j , i ] = Endpoint.ARROW.value
                #     continue
                # elif pk[i][j] == 2: #Ignore PK --> edges
                #     continue
                # elif pk[i][j] == 3: #Ignore PK --> edges
                #     continue
                if ((G.graph[j, i] == Endpoint.TAIL.value and G.graph[i, j] == Endpoint.TAIL.value)
                            or G.graph[j, i] == Endpoint.ARROW.value):  # if Xi - Xj or Xi -> Xj
                        print("----------Yes inside the if block and current i and j:---------", i, j)
                        Hj = np.intersect1d(np.where(G.graph[:, j] == Endpoint.TAIL.value)[0],
                                            np.where(G.graph[j, :] == Endpoint.TAIL.value)[0])  # neighbors of Xj
                        Hi = np.union1d(np.where(G.graph[i, :] != Endpoint.NULL.value)[0],
                                        np.where(G.graph[:, i] != Endpoint.NULL.value)[0])  # adjacent to Xi
                        H0 = np.intersect1d(Hj, Hi)  # find the neighbours of Xj that are adjacent to Xi
                        # for any subset of H0
                        sub = Combinatorial(H0.tolist())  # find all the subsets for H0
                        S = np.ones(len(sub))  # S indicate whether we need to check sub{k}.
                        # 1: check the condition,
                        # 2: check nothing and is valid;
                        for k in range(len(sub)):
                            if (S[k] == 1):
                                V = delete_validity_test(G, i, j, sub[k])  # Delete operator validation test
                                if (V):
                                    # find those subsets that include sub(k)
                                    Idx = find_subset_include(sub[k], sub)
                                    S[np.where(Idx == 1)] = 2  # and set their S to 2
                            else:
                                V = 1

                            if (V):
                                chscore, desc, record_local_score = delete_changed_score(X, G, i, j, sub[k],
                                                                                        record_local_score, score_func,
                                                                                        parameters)
                                # calculate the changed score after Insert operator
                                # desc{count} saves the corresponding (i,j,sub{k})
                                if (chscore < min_chscore):
                                    min_chscore = chscore
                                    min_desc = desc

        if (pk[i][j] == 1 and G.graph[ i , j ] != Endpoint.TAIL.value and G.graph[ j , i ] != Endpoint.ARROW.value):
            p = p + 1
            penalty = weight_lambda*log(p)
            print("----------Current Penalty is:---------", penalty)

        else:
            penalty = weight_lambda*1
            #print("----------Current Penalty for K-edges is:---------", penalty)

        # if (pk[i][j] == 2 and G.graph[ i , j ] != Endpoint.TAIL.value and G.graph[ j , i ] != Endpoint.TAIL.value):
        #     p = p + 1
        #     penalty = weight_lambda*log(p)
        #     print("----------Current Penalty is:---------", penalty)

        # else:
        #     penalty = weight_lambda*1
        #     #print("----------Current Penalty for U-edges is:---------", penalty)

        # if (pk[i][j] == 3 and G.graph[ i , j ] != Endpoint.NULL.value and G.graph[ j , i ] != Endpoint.NULL.value):
        #     p = p + 1
        #     penalty = weight_lambda*log(p)
        #     print("----------Current Penalty is:---------", penalty)

        # else:
        #     penalty = weight_lambda*1
        #     #print("----------Current Penalty for F-edges is:---------", penalty)

        if len(min_desc) != 0:
            score_new = score + min_chscore + penalty
            if score - score_new <= 0:
                break
            G = delete(G, min_desc[0], min_desc[1], min_desc[2])
            update2.append([min_desc[0], min_desc[1], min_desc[2]])
            G = pdag2dag(G)
            G = dag2cpdag(G)
            G_step2.append(G)

        else:
            score_new = score
            break

    Record = {'update1': update1, 'update2': update2, 'G_step1': G_step1, 'G_step2': G_step2, 'G': G, 'score': score}

    print(penalty)

    return Record










# #---------Causal-Learn-GES Code Starts Here----------------

# from typing import Optional
# from causallearn.score.LocalScoreFunctionClass import LocalScoreClass
# from causallearn.graph.GeneralGraph import GeneralGraph
# from causallearn.graph.GraphNode import GraphNode
# from causallearn.utils.DAG2CPDAG import dag2cpdag
# from causallearn.utils.GESUtils import *
# from causallearn.utils.PDAG2DAG import pdag2dag
# from typing import Union


# def ges(X: ndarray, score_func: str = 'local_score_BIC', maxP: Optional[float] = None,
#         parameters: Optional[Dict[str, Any]] = None, node_names: Union[List[str], None] = None,) -> Dict[str, Any]:
#     """
#     Perform greedy equivalence search (GES) algorithm

#     Parameters
#     ----------
#     X : data set (numpy ndarray), shape (n_samples, n_features). The input data, where n_samples is the number of samples and n_features is the number of features.
#     score_func : the string name of score function. (str(one of 'local_score_CV_general', 'local_score_marginal_general',
#                     'local_score_CV_multi', 'local_score_marginal_multi', 'local_score_BIC', 'local_score_BDeu')).
#     maxP : allowed maximum number of parents when searching the graph
#     parameters : when using CV likelihood,
#                   parameters['kfold']: k-fold cross validation
#                   parameters['lambda']: regularization parameter
#                   parameters['dlabel']: for variables with multi-dimensions,
#                                indicate which dimensions belong to the i-th variable.

#     Returns
#     -------
#     Record['G']: learned causal graph, where Record['G'].graph[j,i]=1 and Record['G'].graph[i,j]=-1 indicates  i --> j ,
#                     Record['G'].graph[i,j] = Record['G'].graph[j,i] = -1 indicates i --- j.
#     Record['update1']: each update (Insert operator) in the forward step
#     Record['update2']: each update (Delete operator) in the backward step
#     Record['G_step1']: learned graph at each step in the forward step
#     Record['G_step2']: learned graph at each step in the backward step
#     Record['score']: the score of the learned graph
#     """

#     if X.shape[0] < X.shape[1]:
#         warnings.warn("The number of features is much larger than the sample size!")

#     X = np.mat(X)
#     if score_func == 'local_score_CV_general':  # % k-fold negative cross validated likelihood based on regression in RKHS
#         if parameters is None:
#             parameters = {'kfold': 10,  # 10 fold cross validation
#                           'lambda': 0.01}  # regularization parameter
#         if maxP is None:
#             maxP = X.shape[1] / 2  # maximum number of parents
#         N = X.shape[1]  # number of variables
#         localScoreClass = LocalScoreClass(data=X, local_score_fun=local_score_cv_general, parameters=parameters)

#     elif score_func == 'local_score_marginal_general':  # negative marginal likelihood based on regression in RKHS
#         parameters = {}
#         if maxP is None:
#             maxP = X.shape[1] / 2  # maximum number of parents
#         N = X.shape[1]  # number of variables
#         localScoreClass = LocalScoreClass(data=X, local_score_fun=local_score_marginal_general, parameters=parameters)

#     elif score_func == 'local_score_CV_multi':  # k-fold negative cross validated likelihood based on regression in RKHS
#         # for data with multi-variate dimensions
#         if parameters is None:
#             parameters = {'kfold': 10, 'lambda': 0.01, 'dlabel': {}}  # regularization parameter
#             for i in range(X.shape[1]):
#                 parameters['dlabel'][i] = i
#         if maxP is None:
#             maxP = len(parameters['dlabel']) / 2
#         N = len(parameters['dlabel'])
#         localScoreClass = LocalScoreClass(data=X, local_score_fun=local_score_cv_multi, parameters=parameters)

#     elif score_func == 'local_score_marginal_multi':  # negative marginal likelihood based on regression in RKHS
#         # for data with multi-variate dimensions
#         if parameters is None:
#             parameters = {'dlabel': {}}
#             for i in range(X.shape[1]):
#                 parameters['dlabel'][i] = i
#         if maxP is None:
#             maxP = len(parameters['dlabel']) / 2
#         N = len(parameters['dlabel'])
#         localScoreClass = LocalScoreClass(data=X, local_score_fun=local_score_marginal_multi, parameters=parameters)

#     elif score_func == 'local_score_BIC' or score_func == 'local_score_BIC_from_cov':  # Greedy equivalence search with BIC score
#         if maxP is None:
#             maxP = X.shape[1] / 2
#         N = X.shape[1]  # number of variables
#         parameters = {}
#         parameters["lambda_value"] = 2
#         localScoreClass = LocalScoreClass(data=X, local_score_fun=local_score_BIC, parameters=parameters)

#     elif score_func == 'local_score_BDeu':  # Greedy equivalence search with BDeu score
#         if maxP is None:
#             maxP = X.shape[1] / 2
#         N = X.shape[1]  # number of variables
#         localScoreClass = LocalScoreClass(data=X, local_score_fun=local_score_BDeu, parameters=None)

#     else:
#         raise Exception('Unknown function!')
#     score_func = localScoreClass

#     if node_names is None:
#         node_names = [("X%d" % (i + 1)) for i in range(N)]
#     nodes = []

#     for name in node_names:
#         node = GraphNode(name)
#         nodes.append(node)

#     G = GeneralGraph(nodes)
#     # G = np.matlib.zeros((N, N)) # initialize the graph structure
#     score = score_g(X, G, score_func, parameters)  # initialize the score

#     G = pdag2dag(G)
#     G = dag2cpdag(G)

#     ## --------------------------------------------------------------------
#     ## forward greedy search
#     record_local_score = [[] for i in range(
#         N)]  # record the local score calculated each time. Thus when we transition to the second phase,
#     # many of the operators can be scored without an explicit call the the scoring function
#     # record_local_score{trial}{j} record the local scores when Xj as a parent
#     score_new = score
#     count1 = 0
#     update1 = []
#     G_step1 = []
#     score_record1 = []
#     graph_record1 = []
#     while True:
#         count1 = count1 + 1
#         score = score_new
#         score_record1.append(score)
#         graph_record1.append(G)
#         min_chscore = 1e7
#         min_desc = []
#         for i in range(N):
#             for j in range(N):
#                 if (G.graph[i, j] == Endpoint.NULL.value and G.graph[j, i] == Endpoint.NULL.value
#                         and i != j and len(np.where(G.graph[j, :] == Endpoint.ARROW.value)[0]) <= maxP):
#                     print("YES inside the if block")
#                     # find a pair (Xi, Xj) that is not adjacent in the current graph , and restrict the number of parents
#                     Tj = np.intersect1d(np.where(G.graph[:, j] == Endpoint.TAIL.value)[0],
#                                         np.where(G.graph[j, :] == Endpoint.TAIL.value)[0])  # neighbors of Xj

#                     Ti = np.union1d(np.where(G.graph[:, i] != Endpoint.NULL.value)[0],
#                                     np.where(G.graph[i, :] != Endpoint.NULL.value)[0])  # adjacent to Xi

#                     NTi = np.setdiff1d(np.arange(N), Ti)
#                     T0 = np.intersect1d(Tj, NTi)  # find the neighbours of Xj that are not adjacent to Xi
#                     # for any subset of T0
#                     sub = Combinatorial(T0.tolist())  # find all the subsets for T0
#                     S = np.zeros(len(sub))
#                     # S indicate whether we need to check sub{k}.
#                     # 0: check both conditions.
#                     # 1: only check the first condition
#                     # 2: check nothing and is not valid.
#                     for k in range(len(sub)):
#                         if (S[k] < 2):  # S indicate whether we need to check subset(k)
#                             V1 = insert_validity_test1(G, i, j, sub[k])  # Insert operator validation test:condition 1
#                             if (V1):
#                                 if (not S[k]):
#                                     V2 = insert_validity_test2(G, i, j,
#                                                                sub[k])  # Insert operator validation test:condition 2
#                                 else:
#                                     V2 = 1
#                                 if (V2):
#                                     Idx = find_subset_include(sub[k], sub)  # find those subsets that include sub(k)
#                                     S[np.where(Idx == 1)] = 1
#                                     chscore, desc, record_local_score = insert_changed_score(X, G, i, j, sub[k],
#                                                                                              record_local_score,
#                                                                                              score_func,
#                                                                                              parameters)
#                                     # calculate the changed score after Insert operator
#                                     # desc{count} saves the corresponding (i,j,sub{k})
#                                     # sub{k}:
#                                     if (chscore < min_chscore):
#                                         min_chscore = chscore
#                                         min_desc = desc
#                             else:
#                                 Idx = find_subset_include(sub[k], sub)  # find those subsets that include sub(k)
#                                 S[np.where(Idx == 1)] = 2

#         if (len(min_desc) != 0):
#             score_new = score + min_chscore
#             if (score - score_new <= 0):
#                 break
#             G = insert(G, min_desc[0], min_desc[1], min_desc[2])
#             update1.append([min_desc[0], min_desc[1], min_desc[2]])
#             G = pdag2dag(G)
#             G = dag2cpdag(G)
#             G_step1.append(G)
#         else:
#             score_new = score
#             break

#     ## --------------------------------------------------------------------
#     #counter = 0
#     # backward greedy search
#     count2 = 0
#     score_new = score
#     update2 = []
#     G_step2 = []
#     score_record2 = []
#     graph_record2 = []
#     while True:
#     #while counter < 2:
#         count2 = count2 + 1
#         score = score_new
#         score_record2.append(score)
#         graph_record2.append(G)
#         min_chscore = 1e7
#         min_desc = []
#         for i in range(N):
#             for j in range(N):
#                 if ((G.graph[j, i] == Endpoint.TAIL.value and G.graph[i, j] == Endpoint.TAIL.value)
#                         or G.graph[j, i] == Endpoint.ARROW.value):  # if Xi - Xj or Xi -> Xj
#                     print("----------Backward-search: Yes inside the if block and current i and j:---------", i, j)
#                     Hj = np.intersect1d(np.where(G.graph[:, j] == Endpoint.TAIL.value)[0],
#                                         np.where(G.graph[j, :] == Endpoint.TAIL.value)[0])  # neighbors of Xj
#                     Hi = np.union1d(np.where(G.graph[i, :] != Endpoint.NULL.value)[0],
#                                     np.where(G.graph[:, i] != Endpoint.NULL.value)[0])  # adjacent to Xi
#                     H0 = np.intersect1d(Hj, Hi)  # find the neighbours of Xj that are adjacent to Xi
#                     # for any subset of H0
#                     sub = Combinatorial(H0.tolist())  # find all the subsets for H0
#                     S = np.ones(len(sub))  # S indicate whether we need to check sub{k}.
#                     # 1: check the condition,
#                     # 2: check nothing and is valid;
#                     for k in range(len(sub)):
#                         if (S[k] == 1):
#                             V = delete_validity_test(G, i, j, sub[k])  # Delete operator validation test
#                             if (V):
#                                 # find those subsets that include sub(k)
#                                 Idx = find_subset_include(sub[k], sub)
#                                 S[np.where(Idx == 1)] = 2  # and set their S to 2
#                         else:
#                             V = 1

#                         if (V):
#                             chscore, desc, record_local_score = delete_changed_score(X, G, i, j, sub[k],
#                                                                                      record_local_score, score_func,
#                                                                                      parameters)
#                             # calculate the changed score after Insert operator
#                             # desc{count} saves the corresponding (i,j,sub{k})
#                             if (chscore < min_chscore):
#                                 min_chscore = chscore
#                                 min_desc = desc

#         if len(min_desc) != 0:
#             score_new = score + min_chscore
#             if score - score_new <= 0:
#                 break
#             G = delete(G, min_desc[0], min_desc[1], min_desc[2])
#             update2.append([min_desc[0], min_desc[1], min_desc[2]])
#             G = pdag2dag(G)
#             G = dag2cpdag(G)
#             G_step2.append(G)
#             #counter += 1
#         else:
#             score_new = score
#             break

#     Record = {'update1': update1, 'update2': update2, 'G_step1': G_step1, 'G_step2': G_step2, 'G': G, 'score': score}
#     return Record