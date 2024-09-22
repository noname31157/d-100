import numpy as np
from causallearn.search.ScoreBased.GES import ges
from causallearn.score.LocalScoreFunction import local_score_BIC



dataset = np.load('OT_All_Samples.npy')

Record = ges(dataset, 'local_score_BIC', None, None)

# Visualization using pydot
from causallearn.utils.GraphUtils import GraphUtils
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import io

pyd = GraphUtils.to_pydot(Record['G'], labels = ['age', 'gender', 'bmi', 'surgery', 'trauma', 'medical', 'apsiii', 'sofa', 'smoker', 'copd', 
              'ischemicHd', 'ards', 'death', 'oxygenation', 'spo2', 'fio2', 'sao2', 'pao2', 'paco2', 'ph',
              'lactate', 'hemoglobin', 'peep', 'vt', 'peakAirPressure', 'minVentVol'])
tmp_png = pyd.create_png(f="png")
fp = io.BytesIO(tmp_png)
img = mpimg.imread(fp, format='png')
plt.axis('off')
plt.imshow(img)
plt.show()


# or save the graph
pyd.write_png('simple_test.png')
