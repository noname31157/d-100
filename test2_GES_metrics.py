import time
st = time.time()

import numpy as np
import hashlib

from causallearn.graph.AdjacencyConfusion import AdjacencyConfusion
from causallearn.graph.ArrowConfusion import ArrowConfusion
from causallearn.graph.SHD import SHD
from causallearn.search.ScoreBased.GES import ges
from causallearn.score.LocalScoreFunction import local_score_BIC
#from causallearn.score.LocalScoreFunction import local_score_BIC_NEW
from causallearn.utils.DAG2CPDAG import dag2cpdag
from causallearn.utils.TXT2GeneralGraph import txt2generalgraph
# import debugpy
# debugpy.connect(('localhost', 5678))

#data_path = "child.txt"
#dataset = np.loadtxt(data_path, skiprows=1)
dataset = np.load('data100.npy')
Record = ges(dataset, 'local_score_BIC', None, None) #max parents set to 2/6/3/3/7 (for CHILD/HEPAR2/SACHS/data10/data40)[3rd value]
est = Record['G']

#Metrics generation for CPDAG

truth_dag = txt2generalgraph("data100.graph.txt")
#truth_dag = txt2generalgraph("child.graph.txt")

truth_cpdag = dag2cpdag(truth_dag)

print("DONE conversion of true DAG to true CPDAG")

#truth_cpdag = truth_dag  #DAG's metric estimation if you do not want to convert it into a CPDAG

adj = AdjacencyConfusion(truth_cpdag, est)
arrow = ArrowConfusion(truth_cpdag, est)

adjTp = adj.get_adj_tp()
adjFp = adj.get_adj_fp()
adjFn = adj.get_adj_fn()
adjTn = adj.get_adj_tn()

arrowsTp = arrow.get_arrows_tp()
arrowsFp = arrow.get_arrows_fp()
arrowsFn = arrow.get_arrows_fn()
arrowsTn = arrow.get_arrows_tn()
arrowsTpCE = arrow.get_arrows_tp_ce()
arrowsFpCE = arrow.get_arrows_fp_ce()
arrowsFnCE = arrow.get_arrows_fn_ce()
arrowsTnCE = arrow.get_arrows_tn_ce()

adjPrec = adj.get_adj_precision()
adjRec = adj.get_adj_recall()
adjFDR = adj.get_adj_FDR ()
arrowPrec = arrow.get_arrows_precision()
arrowRec = arrow.get_arrows_recall()
arrowFDR = arrow.get_arrows_FDR()
arrowPrecCE = arrow.get_arrows_precision_ce()
arrowRecCE = arrow.get_arrows_recall_ce()


print(f"ArrowsTp: {arrowsTp}")
print(f"ArrowsFp: {arrowsFp}")
print(f"ArrowsFn: {arrowsFn}")

print(f"AdjTPR: {adjRec}")
print(f"AdjFDR: {adjFDR}")

shd = SHD(truth_cpdag, est)
print(f"SHD: {shd.get_shd()}")
print(f"TPR(arrow): {arrowRec}")
print(f"FDR(arrow): {arrowFDR}")


# et = time.time()
# elapsed_time = et - st
# print('Execution time:', elapsed_time, 'seconds')

# Visualization using pydot
from causallearn.utils.GraphUtils import GraphUtils
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import io


#pyd = GraphUtils.to_pydot(Record['G'], labels = ['X1','X2','X3','X4','X5','X6','X7'])
#pyd = GraphUtils.to_pydot(Record['G'], labels = ['X1','X2','X3','X4','X5','X6','X7','X8'])
#pyd = GraphUtils.to_pydot(Record['G'], labels = ['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10'])
# pyd = GraphUtils.to_pydot(Record['G'], labels = ['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10'])
#pyd = GraphUtils.to_pydot(Record['G'], labels = ['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11','X12','X13'])
#pyd = GraphUtils.to_pydot(Record['G'], labels = ['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11','X12','X13','X14','X15'])
#pyd = GraphUtils.to_pydot(Record['G'], labels = ['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11','X12','X13','X14','X15','X16','X17','X18','X19','X20'])
#pyd = GraphUtils.to_pydot(Record['G'], labels = ['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11','X12','X13','X14','X15','X16','X17','X18','X19','X20','X21','X22','X23','X24','X25'])
#pyd = GraphUtils.to_pydot(Record['G'], labels = ['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11','X12','X13','X14','X15','X16','X17','X18','X19','X20','X21','X22','X23','X24','X25','X26','X27','X28','X29','X30','X31','X32','X33','X34','X35','X36','X37'])
#pyd = GraphUtils.to_pydot(Record['G'], labels = ['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11','X12','X13','X14','X15','X16','X17','X18','X19','X20','X21','X22','X23','X24','X25','X26','X27','X28','X29','X30','X31','X32','X33','X34','X35','X36','X37','X38','X39','X40'])
pyd = GraphUtils.to_pydot(Record['G'], labels = ['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11','X12','X13','X14','X15','X16','X17','X18','X19','X20','X21','X22','X23','X24','X25','X26','X27','X28','X29','X30','X31','X32','X33','X34','X35','X36','X37','X38','X39','X40','X41','X42','X43','X44','X45','X46','X47','X48','X49','X50','X51','X52','X53','X54','X55','X56','X57','X58','X59','X60','X61','X62','X63','X64','X65','X66','X67','X68','X69','X70','X71','X72','X73','X74','X75','X76','X77','X78','X79','X80','X81','X82','X83','X84','X85','X86','X87','X88','X89','X90','X91','X92','X93','X94','X95','X96','X97','X98','X99','X100'])


#OT_GT
#pyd = GraphUtils.to_pydot(Record['G'], labels = ['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11','X12','X13','X14','X15','X16','X17','X18','X19','X20','X21','X22','X23','X24','X25','X26'])


tmp_png = pyd.create_png(f="png")
fp = io.BytesIO(tmp_png)
img = mpimg.imread(fp, format='png')
plt.axis('off')
plt.imshow(img)
plt.show()









