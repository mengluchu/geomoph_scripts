#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 10:07:06 2021

@author: menglu
"""

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
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn import tree
from io import StringIO
import pydotplus
from IPython.display import Image
import verstack
from verstack.stratified_continuous_split import scsplit
  
  train, test = scsplit(data, stratify = data['continuous_column_name'])
  X_train, X_val, y_train, y_val = scsplit(X, y, stratify = y, 
                                           test_size = 0.3, random_state = 5)
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
recall_all = np.zeros(1)
precision_all= np.zeros(1)  
filename = "/Volumes/Meng_Mac/obia/temp/HP1103.gpkg"
resultdir2 = "/Users/menglu/Volumes/Meng_Mac/obia/temp/" # for saving precision and recall npy.
treedir = "/Users/menglu/Downloads"
layers = fiona.listlayers(f'{filename}') 
layers
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
def getresults(Xtest, clf, threshold,  yhat_labels=None, graphdir=f'{treedir}/xgb_tree{seeds}.png'):
                if yhat_labels is None: 
                    yhat = clf.predict(X_test)
                     
                    # threshold 0.5, probability higher than 0.5 -> positive. 
                    yhat_labels = yhat>threshold
                    yhat_labels = yhat_labels.astype(int)
                if graphdir is not None: #save graph
                    out = StringIO()
                    tree.export_graphviz(clf, out_file=out, class_names=['0','1'], feature_names =list(X_test.columns),filled=True, rounded=True, special_characters=True)
        
                    graph=pydotplus.graph_from_dot_data(out.getvalue())
                    with open(graphdir, "wb") as png:
                        png.write(graph.create_png())
                #with open(f'{treedir}/xgb_tree{seeds}_noxgb.png', "wb") as png:
                #     png.write(graph.create_png())
                #Image(graph.create_png())
    
                #get accuracy score
                accuracy = accuracy_score(Y_testnum, yhat_labels)
                # get precision and recall
                # print("precision:  tp / (tp + fp)")
                # print(label_all)
                recall=np.round(recall_score(Y_testnum, yhat_labels, average = None),2)[0]
                # only get the recall and precision for the class of interest, therefore "[0]"
                precision = np.round(precision_score(Y_testnum, yhat_labels, average = None),2)[0]
                print(accuracy, recall, precision)
                return(accuracy, recall, precision)
            
def run(k, j, filename, seednum=10, threshold = 0.5, resultdir=None, graphdir = f'{treedir}/'):
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
        avepre = np.zeros(1) # store all the xgb+tree precisions in each CV
        averec = np.zeros(1)
        avepre2 = np.zeros(1) # store all the tree precisions in each CV
        averec2 = np.zeros(1)
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
            
            # ZGB model on training! for fitting a tree
            yhat = model.predict(dtrain)
            yhat_labels = yhat>threshold
            yhat_labels = yhat_labels.astype(int)
            
            yhat2 = model.predict(dtest) #comparison purpose
            yhat_labels2 = yhat2>threshold
            yhat_labels2 = yhat_labels2.astype(int)
            #print(yhat_labels2.shape)
            # threshold 0.5, probability higher than 0.5 -> positive. 
            
            
            # fit a single tree to XGBoost predictions. 
            clf = DecisionTreeClassifier(min_samples_split= 30, max_depth= 4, min_samples_leaf=20, random_state=1)
            # note random state is only for max_feature < n_feature, so not useful here.
            clf = clf.fit(X_train, yhat_labels)   
           
            # evaluation after fiting a single tree. 
            # yhat_labels: can be given, for only using xgb, otherwise predicts from clf
            # clf: classification model fit 
            
            graphdir = f'{graphdir}treegraph{seeds}.png'
            accuracy, recall, precision = getresults(Xtest = X_test, clf =clf, threshold=threshold, yhat_labels=None, graphdir = graphdir)
            averec = np.append(averec, recall) #store all of them
            avepre= np.append(avepre, precision)
             # if only fitting a tree
            clf = clf.fit(X_train, Y_trainnum) # for comparison purpose: fitting a tree directly  
            #only xgb
            #accuracy, recall, precision = getresults(Xtest = X_test, clf =clf, threshold=threshold, yhat_labels=yhat_labels2)
            
            #only tree
            accuracy, recall, precision = getresults(Xtest = X_test, clf =clf, threshold=threshold, yhat_labels=None, graphdir = graphdir)

            averec2 = np.append(averec, recall) #store all of them
            avepre2= np.append(avepre, precision)

        recall = averec.sum()/seednum #get the mean but exclude the first one (0)
        precision = avepre.sum()/seednum
        recall2 = averec2.sum()/seednum #get the mean but exclude the first one (0)
        precision2 = avepre2.sum()/seednum
        print(averec, recall)
        if resultdir is not None:
            Y_testnum =  Y_testnum.astype(int)
            plt.rcParams.update({'font.size': 8})
            ax = xgb.plot_importance(model, grid=False, importance_type='gain', title='Feature importance')
            ax.set_title(f'xgboost importance {layers[j]} {classes[k]}')
            fname = f"{resultdir}P_{layers[j]}_{classes[k]}_imp"
            plt.savefig(fname, dpi=1200)
        return (recall, precision, recall2, precision2)
 
#filedir = "/Users/menglu/Documents/GitHub/waaden/OBIA_RF/results" # for confusion matrices
#resultdir = "/Users/menglu/Volumes/Meng_Mac/obia/results/" # for var imp plots
recall_xgbtree, precision_xgbtree, recall_tree, precision_tree = run(k = 1,j = 2, filename = filename, seednum=10, threshold = 0.5, graphdir =f'{treedir}/xgb_tree{seeds}.png' )
print(recall_xgbtree, precision_xgbtree, recall_tree, precision_tree)
