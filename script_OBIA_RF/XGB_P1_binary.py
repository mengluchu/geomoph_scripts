#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 16:34:49 2021

@author: menglu
"""

# %% [code]
import os 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import geopandas as gp
import pandas as pd
import sklearn 
from sklearn.ensemble import RandomForestClassifier
import xgboost
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import xgboost as xgb
def cl2idx(inputarr, dict_label):
    y_label = np.zeros((inputarr.shape[0] ))
    for i in range(inputarr.shape[0]):
            y_label[i] = dict_label[inputarr[i, 0]]
    return y_label



# %% [code]
np.random.seed(333)
joind = gp.read_file("../input/reclassified-ml/reclasified_ml/reclassified_ml.shp")
classes = ['P1a1', 'P1a2'] 

df1 = pd.DataFrame(joind.drop(columns='geometry'))
df1 = df1.replace([np.inf, -np.inf], np.nan).dropna()

df1.columns 

# %% [code]
classes

# %% [code]
Pcl = df1.loc[df1['group_M'].isin(classes)]

# %% [code]
print(f'total {len(df1)}, P_classes: {len(Pcl)}')

# %% [code]
len(Pcl)/2*0.7

# %% [code]
train = Pcl.groupby('group_M').sample(n =10000).index
test = Pcl[~Pcl.index.isin(train)].index
print(len(train),len(test))

# %% [code]
df_covar = Pcl.filter(regex='density|Dif_|EF|GLCM|LW|max_SI|mean_|mn_|num|obj|SI_|std|tot')
df_covar[df_covar>1e5] =0 

# %% [code]
 
X_train = df_covar.loc [train ]
X_test  = df_covar.loc  [test ]
Y_train  =Pcl.filter(regex='group_M').loc[train ]
Y_test  =Pcl.filter(regex='group_M').loc[test ]
Y_train= Y_train.values
Y_test = Y_test.values
 


 #string.ascii_lowercase

"""xgboost binary classification"""


# %% [code]
label_all =  Pcl.group_M.unique()
i = 0
idx2class = {}
class2idx = {}
for tp in label_all:
    idx2class[i] = tp
    class2idx[tp] = i 
    i+= 1


# %% [code]
 
Y_trainnum =  cl2idx(Y_train, class2idx)
Y_testnum =  cl2idx(Y_test, class2idx)
 
 

# %% [code]
Y_train.shape

# %% [code]
np.unique(Y_trainnum)

# %% [code]

dtrain = xgb.DMatrix(X_train, label=Y_trainnum)
dtest = xgb.DMatrix(X_test, label=Y_testnum)
#params = {'max_depth': 6, 'eta': 0.002, 'silent': 1, 'n_estimators' : 1500,
#          'objective': 'multi:softprob', 'num_class': len(np.unique(Y_trainnum)), 'eval_metric':['merror', 'mlogloss', 'auc' ] }
# Fit
params = {'max_depth': 6, 'eta': 0.002, 'silent': 1, 'n_estimators' : 1500,
          'objective': 'binary:logistic',  'eval_metric':['merror', 'mlogloss', 'auc' ] }

# Evalute

 
 

# %% [code]
model = xgb.train(params, dtrain, 100)

# %% [code]
yhat = model.predict(dtest)
yhat_labels =  yhat>0.5
yhat_labels =yhat_labels.astype(int)

# %% [code]
np.amin(yhat_labels)

# %% [code]

TPTN = sum(yhat_labels ==Y_testnum)

# %% [code]

FNFP = sum(yhat_labels !=Y_testnum)

# %% [code]
FPTP = sum(yhat_labels==1 )

# %% [code]
accuracy = accuracy_score(Y_testnum, yhat_labels)
print("precision:  tp / (tp + fp)")
print(precision_score(Y_testnum, yhat_labels, average = None))

print("recall:  tp / (tp + fn)") 
print(recall_score(Y_testnum, yhat_labels, average = None))

recall=recall_score(Y_testnum, yhat_labels, average = 'weighted')
precision=precision_score(Y_testnum, yhat_labels, average = "weighted")
print("Accuracy: %.2f%%" % (accuracy * 100.0))
print("Precision: %.2f%% " % (precision *100))
print("Recall: %.2f%% " % (recall * 100))
print(class2idx)




# %% [code]
Y_testnum =  Y_testnum.astype(int)

# %% [code]
cm  = confusion_matrix(Y_testnum, yhat_labels)
cm

# %% [code]
plt.rcParams.update({'font.size': 4})
xgb.plot_importance(model, grid=False, importance_type='gain', title='Feature importance')

fname = "obia importance"
plt.savefig(fname, dpi=1200)

 

# %% [code]
'''
plt.rcParams.update({'font.size': 4})
xgb.plot_tree(model, num_trees=2)
fname = "xgb_tree2"
plt.savefig(fname, dpi=1200)

plt.rcParams.update({'font.size': 4})
xgb.plot_tree(model, num_trees=1000)
fname = "xgb_tree1000"
plt.savefig(fname, dpi=1200)
'''


# %% [code]
#xgb.to_graphviz(model, num_trees=2)

# %% [code]
# column is predicted
plt.rcParams.update({'font.size': 11}) 
fig, ax = plt.subplots(1,1)

plt.imshow(cm, cmap=plt.get_cmap("GnBu"))
ax.set_xticks([0,1])
ax.set_yticks([0,1])
ax.set_xticklabels(classes )
ax.set_yticklabels(classes )

plt.colorbar()

# %% [code]
fig, ax = plt.subplots(1,1)
cm  = confusion_matrix(Y_testnum, yhat_labels,normalize='true')
plt.imshow(cm, cmap=plt.get_cmap("GnBu"))
ax.set_xticks([0,1])
ax.set_yticks([0,1])
ax.set_xticklabels(classes )
ax.set_yticklabels(classes )

plt.colorbar() 
# TP/ all True

# %% [code]
fig, ax = plt.subplots(1,1)
cm  = confusion_matrix(Y_testnum, yhat_labels,normalize='pred')
plt.imshow(cm, cmap=plt.get_cmap("GnBu"))
ax.set_xticks([0,1])
ax.set_yticks([0,1])
ax.set_xticklabels(classes )
ax.set_yticklabels(classes )

plt.colorbar() 
# TP/ (all predicted T), most of the times, the predictions. 

# %% [code]
testgeo = pd.DataFrame(joind["geometry"].iloc[test])
testgeo["test_truth"] =  Y_test
testgeo["predicted"] =  yhat_labels
testgeo["truth_num"] = Y_testnum
testgeo = testgeo.set_geometry('geometry')

# %% [code]
 
testgeo.to_file("test_P1.shp")
 

# %% [code]


def merror(predt: np.ndarray, dtrain: xgb.DMatrix):
    y = dtrain.get_label()
    # Like custom objective, the predt is untransformed leaf weight
    assert predt.shape == (kRows, kClasses)
    out = np.zeros(kRows)
    for r in range(predt.shape[0]):
        i = np.argmax(predt[r])
        out[r] = i

    assert y.shape == out.shape

    errors = np.zeros(kRows)
    errors[y != out] = 1.0
    return 'PyMError', np.sum(errors) / kRows

