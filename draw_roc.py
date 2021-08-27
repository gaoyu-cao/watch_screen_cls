'''
Descripttion: 
version: 
Author: 
Date: 2021-08-05 03:06:57
LastEditors: Please set LastEditors
LastEditTime: 2021-08-05 03:06:58
'''
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc 


fpr = [0.08727707482315433, 0.045332270598784495,0.03815881239414168, 0.03377503238019328,
            0.03168277373717246, 0.02839493872671117,0.02690046826741058,0.025107103716249874,
            0.0221181627976487,0.02042442961044137,0.011557238218591212,0.0051808309255753715,0.002490784098834313,
            0.0008966822755803527]

tpr = [0.8204503956177723, 0.7766281192939745, 0.7626293365794279,0.7541083384053561,0.7461959829580036,
0.740718198417529,0.7334144856968959,0.7255021302495435,0.7133292757151553,0.6932440657334145,0.6068167985392574,0.4625684723067559,
0.3213633597078515, 0.18807060255629945]
roc_auc = auc(fpr,tpr)

plt.figure()
plt.title('ROC Curve',fontweight='bold')
plt.plot(fpr, tpr, color='darkorange',
             lw=2, label='ROC curve (area = %0.3f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
plt.xlabel('False Positivve Rate',fontweight='bold')
plt.ylabel('True Positivve Rate',fontweight='bold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.savefig('./roc.png')