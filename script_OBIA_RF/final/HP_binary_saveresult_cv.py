#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 21:04:00 2021

@author: menglu
"""
 
# python3.6 -W ignore ~/Documents/GitHub/waaden/OBIA_RF/script_OBIA_RF/four_P_binary.py > P_subv1.txt

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
#import warnings
#warnings.filterwarnings("ignore", category=FutureWarning)

#class to index
def cl2idx(inputarr, dict_label):
    y_label = np.zeros((inputarr.shape[0] ))
    for i in range(inputarr.shape[0]):
            y_label[i] = dict_label[inputarr[i, 0]]
    return y_label

#layers = fiona.listlayers("/Volumes/Meng_Mac/obia/temp/results2.gdb") # feb results
#layers = fiona.listlayers("/Volumes/Meng_Mac/obia/temp/results_0303.gdb") # better water 
#layers = fiona.listlayers("/Volumes/Meng_Mac/obia/temp/P_subv1_1.gdb") # better water 
#layers = fiona.listlayers("/Volumes/Meng_Mac/obia/temp/HP1103.gpkg") # better water 
#layers
#filename = "/Volumes/Meng_Mac/obia/temp/HP1103.gpkg"

''' run 
# params
# j: iterate over layers
# k: iterate over classes 
# filename: directory of a file that can be read by geopandas
# seednum: number of CV iterations 
# threshold: classify as positive if higher than this probability
# resultdir: dir to save variable importance plots. if not provided, then varimp are not calcuated

# return
# two numbers, recall and precision
# recall and precision are -1 if not calculated due to too few objects.  
'''

def run(k, j, filename, seednum=10, threshold = 0.5, resultdir=None):
#    classes = ["P1a1" , "P1a2"  , "P2b"  , "P2c" ] 
    classes = ["P1a1" , "P1a2", "P2b", "P2c", "H1" ]
    # H1 H2  O (1) P1a1 (4)  P1a2 (6)   P2b   P2c   S1a (0)   S1c    S2    S3 
    joind = gp.read_file(filename, layer = layers[j])
    print(f'\n------\n------{layers[j]}----\n-----\n')
    df1 = pd.DataFrame(joind.drop(columns='geometry'))
    df1 = df1.replace([np.inf, -np.inf], np.nan).dropna()
    
    Pcl = df1.loc[df1['geocode_2'].isin(classes)] # filter only classes of interest
    print(Pcl['geocode_2'].value_counts())
    # regroup, geocode_2 from here on becomes binary!
    Pcl['geocode_2'] = np.where(Pcl['geocode_2'].str.contains(classes[k]),classes[k],'Others')
    print(Pcl['geocode_2'].value_counts())
    minc = min(Pcl['geocode_2'].value_counts() ) # skip if less than 20 objects 
    if minc< 20 or minc==len(Pcl):
        print("minimum class less than 20")
        return (-1, -1) # -1 -1 if not calculated
    else:    
        print(f'total {len(df1)}, P_H1_classes: {len(Pcl)}, minimun class: {minc}')       
        # bootstrap and get averaged accuracy
        avepre = np.zeros(1) # store all the precisions in each CV
        averec = np.zeros(1)
        for seeds in range(seednum):
            np.random.seed(seeds)
            # use groupby to sample the same amount for each group. 
            # use 70% of data for training, get the index
            train = Pcl.groupby('geocode_2').sample(n = int(minc*0.7)).index 
            test = Pcl[~Pcl.index.isin(train)].index
            #len(train)+len(test)
             
            df_covar = Pcl
            X_train = df_covar.loc [train ].drop(columns=["geocode_2","layer","OBJECTID","path"])
            X_test  = df_covar.loc [test ].drop(columns=["geocode_2","layer","OBJECTID","path"])
            
            Y_train  =Pcl.filter(regex='geocode_2').loc[train].values
            Y_test  =Pcl.filter(regex='geocode_2').loc[test].values
            
            # relable
            label_all = [classes[k], "Others"]
            #classtype  =  [(j, "float32") for j in classes]
            
            #Pcl.geocode_2.unique()
            i = 0
            idx2class = {}
            class2idx = {}
            for tp in label_all:
                idx2class[i] = tp
                class2idx[tp] = i 
                i+= 1
            
            Y_trainnum = cl2idx(Y_train, class2idx).astype(int)
            Y_testnum = cl2idx(Y_test, class2idx).astype(int)
             
            np.unique(Y_trainnum)
            # can consider use scikitlearn or h2o to replace the xgb API. 
            # note the estimators can only be specified in the xgb.train, not in the params. 
            dtrain = xgb.DMatrix(X_train, label=Y_trainnum)
            dtest = xgb.DMatrix(X_test, label=Y_testnum)
            params = {'max_depth': 6, 'eta': 0.002, 
                      'objective':'binary:logistic', 'num_class': 1, 'eval_metric':['merror', 'mlogloss', 'auc' ] }
            # Fit
            #print("Train and test shapes, dividing number of classes for the sample size (i.e. 2 for binary case)")
            #print(X_train.shape, Y_trainnum.shape, X_test.shape, Y_testnum.shape)
            model = xgb.train(params, dtrain,  500) #numroudnd = 500
 
            yhat = model.predict(dtest)
            # threshold 0.5, probability higher than 0.5 -> positive. 
            yhat_labels = yhat>threshold
            yhat_labels = yhat_labels.astype(int)
            
            #get accuracy score
            accuracy = accuracy_score(Y_testnum, yhat_labels)
            # get precision and recall
            # print("precision:  tp / (tp + fp)")
            # print(label_all)
            recall=np.round(recall_score(Y_testnum, yhat_labels, average = None),2)[0]
            # only get the recall and precision for the class of interest, therefore "[0]"
            precision = np.round(precision_score(Y_testnum, yhat_labels, average = None),2)[0]
            averec = np.append(averec, recall) #store all of them
            avepre= np.append(avepre, precision)
        recall = averec.sum()/seednum #get the mean but exclude the first one (0)
        precision = avepre.sum()/seednum
        print(averec, recall)
        if resultdir is not None:
            Y_testnum =  Y_testnum.astype(int)
            plt.rcParams.update({'font.size': 8})
            ax = xgb.plot_importance(model, grid=False, importance_type='gain', title='Feature importance')
            ax.set_title(f'xgboost importance {layers[j]} {classes[k]}')
            fname = f"{resultdir}P_{layers[j]}_{classes[k]}_imp"
            plt.savefig(fname, dpi=1200)
        return (recall, precision)
 
recall_all = np.zeros(1)
precision_all= np.zeros(1)  
filename = "/Volumes/Meng_Mac/obia/temp/HP1103.gpkg"
resultdir2 = "/Users/menglu/Volumes/Meng_Mac/obia/temp/" # for saving precision and recall npy.

layers = fiona.listlayers(f'{filename}') 
#filedir = "/Users/menglu/Documents/GitHub/waaden/OBIA_RF/results" # for confusion matrices
#resultdir = "/Users/menglu/Volumes/Meng_Mac/obia/results/" # for var imp plots

for k in range(5):
        
    for j in range(len(layers)-1):

        recall, precision = run(k = k,j = j, filename = filename, seednum=10, threshold = 0.5)
        recall_all = np.append(recall_all, recall)
        precision_all= np.append(precision_all, precision)
 
np.save(f'{resultdir2}recall', recall_all)
np.save(f'{resultdir2}precision', precision_all)

''' plot 
plotdir = '/Volumes/Meng_Mac/obia/temp/phconf/'
os.mkdir(f'{plotdir}')
length = len(layers)-1
precision = np.load('/Volumes/Meng_Mac/obia/temp/precision.npy') 
#recall = np.load('/Volumes/Meng_Mac/obia/temp/recall.npy')
precision = precision[1:]
def sep(start, stop, step=1.):
  """
    Like np.linspace but uses step instead of num
    This is inclusive to stop, so if start=1, stop=3, step=0.5
    Output is: array([1., 1.5, 2., 2.5, 3.])
  """
  return np.linspace(start, stop, int((stop - start) / step + 1))

sep1 =  sep(0, len(precision),length)[:-1].astype(int)
sep2 = sep(0, len(precision),length)[1:].astype(int)
# classes = ["P1a1" , "P1a2"  , "P2b"  , "P2c", "H1" ]
x= layers[:-1]
for i in range(5):
    class_n = precision[sep1[i]:sep2[i]] # first class 
    # classes = ["P1a1" , "P1a2"  , "P2b"  , "P2c", "H1" ]
    plt.figure(tight_layout=True)
    plt.plot(class_n) 
    plt.xticks(range(len(class_n)), labels = x, rotation = "vertical")
    plt.title(f'{classes[i]}_Prec')
    plt.savefig(f'{plotdir}{classes[i]}_Prec')
    plt.close("all")
 
'''