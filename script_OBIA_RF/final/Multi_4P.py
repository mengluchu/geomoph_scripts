#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 19:49:16 2021

@author: menglu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 17:29:53 2021

@author: menglu
"""
import fpdf

data=[1,2,3,4,5,6]

pdf = fpdf.FPDF(format='letter')
pdf.add_page()
pdf.set_font("Arial", size=12)


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
import fiona
def cl2idx(inputarr, dict_label):
    y_label = np.zeros((inputarr.shape[0] ))
    for i in range(inputarr.shape[0]):
            y_label[i] = dict_label[inputarr[i, 0]]
    return y_label

filedir = "/Users/menglu/Documents/GitHub/waaden/OBIA_RF/results"
 
layers = fiona.listlayers("/Volumes/Meng_Mac/obia/results2.gdb")
#j=1 
def run(j):
   # joind = gp.read_file("/Volumes/Meng_Mac/obia/results2.gdb", layer = layers[j])
    joind = gp.read_file("/Users/menglu/Downloads/water_test_p2bwater.gdb", layer = layers[j])

    classes = ["P1a1" , "P1a2"  , "P2b"  , "P2c" ] 
    #H1    H2     O (1) P1a1 (4)  P1a2 (6)   P2b   P2c   S1a (0)   S1c    S2    S3 
    df1 = pd.DataFrame(joind.drop(columns='geometry'))
    df1 = df1.replace([np.inf, -np.inf], np.nan).dropna()
    #classes
    
    # %% [code]
    Pcl = df1.loc[df1['geocode_2'].isin(classes)]
    np.unique(df1['geocode_2']) 
    
    minc = min(Pcl['geocode_2'].value_counts() )
    # %% [code]
    print(f'total {len(df1)}, P_classes: {len(Pcl)}, minimun class: {minc}')
    # %% [code]
    np.random.seed(333)
    train = Pcl.groupby('geocode_2').sample(n = int(minc*0.7)).index
    test = Pcl[~Pcl.index.isin(train)].index
    len(train)+len(test)
    
    # %% [code]
    df_covar = Pcl
    #.filter(regex='density|Dif_|EF|GLCM|LW|max_SI|mean_|mn_|num|obj|SI_|std|tot')
    #df_covar[df_covar>1e5] =0 
    
    # %% [code]
     
    X_train = df_covar.loc [train ].drop(columns=["geocode_2"])
    X_test  = df_covar.loc  [test ].drop(columns=["geocode_2"])
    
    Y_train  =Pcl.filter(regex='geocode_2').loc[train ]
    Y_test  =Pcl.filter(regex='geocode_2').loc[test ]
    Y_train= Y_train.values
    Y_test = Y_test.values
     
    """xgboost softmax regression"""
    
    # %% [code]
    label_all =classes
    #classtype  =  [(j, "float32") for j in classes]
    
#    Pcl.geocode_2.unique()
    i = 0
    idx2class = {}
    class2idx = {}
    for tp in label_all:
        idx2class[i] = tp
        class2idx[tp] = i 
        i+= 1
    
    Y_trainnum = cl2idx(Y_train, class2idx)
    Y_testnum = cl2idx(Y_test, class2idx)
     
    np.unique(Y_trainnum)
    X_train.dtypes
    
    dtrain = xgb.DMatrix(X_train, label=Y_trainnum)
    dtest = xgb.DMatrix(X_test, label=Y_testnum)
    params = {'max_depth': 6, 'eta': 0.002, 'silent': 1, 'n_estimators' : 500,
              'objective': 'multi:softprob', 'num_class': len(np.unique(Y_trainnum)), 'eval_metric':['merror', 'mlogloss', 'auc' ] }
    # Fit
    model = xgb.train(params, dtrain, 100)
    
    # %% [code]
    yhat = model.predict(dtest)
    yhat_labels = np.argmax(yhat, axis=1)
    
    # %% [code]
    accuracy = accuracy_score(Y_testnum, yhat_labels)
    print("precision:  tp / (tp + fp)")
    print(np.round(precision_score(Y_testnum, yhat_labels, average = None),2))
    
    print("recall:  tp / (tp + fn)") 
    print(np.round(recall_score(Y_testnum, yhat_labels, average = None),2))
        
    recall=recall_score(Y_testnum, yhat_labels, average = 'weighted')
    precision=precision_score(Y_testnum, yhat_labels, average = "weighted")
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print("Precision: %.2f%% " % (precision *100))
    print("Recall: %.2f%% " % (recall * 100))
    print(class2idx)
    
    
     
    Y_testnum =  Y_testnum.astype(int)
    
     
    plt.rcParams.update({'font.size': 8})
    ax = xgb.plot_importance(model, grid=False, importance_type='gain', title='Feature importance')
    ax.set_title(f'xgboost importance {layers[j]}')
    fname = f"{filedir}/P_{layers[j]}_water_imp"
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
    
    cm  = confusion_matrix(Y_testnum, yhat_labels)  
    print(classes, f'\n',  cm)
    fig, ax = plt.subplots(1,1)
    plt.rcParams.update({'font.size': 12})
    plt.imshow(np.log(cm), cmap=plt.get_cmap("GnBu"))
    
    ax.set_xticks([0,1,2,3])
    ax.set_yticks([0,1,2,3])
    ax.set_xticklabels(classes )
    ax.set_yticklabels(classes )
    
    plt.colorbar()
    fname = f"{filedir}/cmlog_{layers[j]}"
    plt.savefig(fname, dpi=1200)
    # %% [code]
    fig, ax = plt.subplots(1,1)
    cm  = np.round( confusion_matrix(Y_testnum, yhat_labels,normalize='true'),2)
    print(classes, f'\n',  cm)
    plt.imshow(cm, cmap=plt.get_cmap("GnBu"))
    ax.set_xticks([0,1,2,3])
    ax.set_yticks([0,1,2,3])
    ax.set_xticklabels(classes )
    ax.set_yticklabels(classes )
    
    plt.colorbar() 
    fname = f"{filedir}/cm_tp_alltrue_{layers[j]}"
    plt.savefig(fname, dpi=1200)
    # TP/ all True
    
    # %% [code]
    fig, ax = plt.subplots(1,1)
    cm  = np.round(confusion_matrix(Y_testnum, yhat_labels,normalize='pred'),2)
    print(classes, f'\n',  cm)
    plt.imshow(cm, cmap=plt.get_cmap("GnBu"))
    ax.set_xticks([0,1,2,3])
    ax.set_yticks([0,1,2,3])
    ax.set_xticklabels(classes )
    ax.set_yticklabels(classes )
    
    plt.colorbar() 
    fname = f"{filedir}/cm_tp_allpred_{layers[j]}"
    plt.savefig(fname, dpi=1200)
    # TP/ (all predicted T), most of the times, the predictions. 
for j in range(4):
    run(j)





# %% [code]
testgeo = pd.DataFrame(joind["geometry"].iloc[test])
testgeo["test_truth"] =  Y_test
testgeo["predicted"] =  yhat_labels
testgeo["truth_num"] = Y_testnum
testgeo = testgeo.set_geometry('geometry')

# %% [code]
 
testgeo.to_file("testP.shp")
 

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


 
# %% [code]
