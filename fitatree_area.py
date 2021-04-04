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
#import pydotplus
from IPython.display import Image
import verstack
from verstack.stratified_continuous_split import scsplit
import imblearn 
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids
from imblearn.under_sampling import TomekLinks 
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
resultdir = "/Users/menglu/Volumes/Meng_Mac/obia/temp/" # for saving precision and recall npy.
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
def getresults(Xtest,Y_testnum, clf, threshold,seeds,  yhat_labels=None, graphdir=f'{treedir}'):
                if yhat_labels is None: 
                    yhat = clf.predict(Xtest)
                     
                    # threshold 0.5, probability higher than 0.5 -> positive. 
                    yhat_labels = yhat>threshold
                    yhat_labels = yhat_labels.astype(int)
                if graphdir is not None: #save graph
                    out = StringIO()
                    tree.export_graphviz(clf, out_file=out, class_names=['0','1'], feature_names =list(Xtest.columns),filled=True, rounded=True,    special_characters=True)
        
                    graph=pydotplus.graph_from_dot_data(out.getvalue())
                    with open(f'{graphdir}/xgb_tree{seeds}.png', "wb") as png:
                        png.write(graph.create_png())
                #with open(f'{treedir}/xgb_tree{seeds}_noxgb.png', "wb") as png:
                #     png.write(graph.create_png())
                #Image(graph.create_png())
                TP = sum(y_hat == y_pred)
                #get accuracy score
                accuracy = accuracy_score(Y_testnum, yhat_labels)
                # get precision and recall
                # print("precision:  tp / (tp + fp)")
                # print(label_all)
                recall=np.round(recall_score(Y_testnum, yhat_labels, average = None),2)[0]
                # only get the recall and precision for the class of interest, therefore "[0]"
                precision = np.round(precision_score(Y_testnum, yhat_labels, average = None),2)[0]
                print(seeds, accuracy, recall, precision)
                return(accuracy, recall, precision)

def comp_confmat(actual, predicted):

    classes = np.unique(actual) # extract the different classes
    matrix = np.zeros((len(classes), len(classes))) # initialize the confusion matrix with zeros

    for i in range(len(classes)):
        for j in range(len(classes)):

            matrix[i, j] = np.sum((actual == classes[i]) & (predicted == classes[j]))

    return matrix
   
def run(k, j, filename, seednum=20, threshold = 0.5, resultdir=None, graphdir = f'{treedir}/'):
#    classes = ["P1a1" , "P1a2"  , "P2b"  , "P2c" ] 
    classes = ["P1a1" , "P1a2", "P2b", "P2c", "H1" ]
    # H1 H2  O (1) P1a1 (4)  P1a2 (6)   P2b   P2c   S1a (0)   S1c    S2    S3 
    joind = gp.read_file(filename, layer = layers[j])
    print(f'\n------\n------{layers[j]}----\n-----\n')
    joind['area']= joind['geometry'].area #calculate the area of each object
    df1 = pd.DataFrame(joind.drop(columns='geometry'))
    df1 = df1.replace([np.inf, -np.inf], np.nan).dropna()
    
    Pcl = df1.loc[df1['geocode_2'].isin(classes)] # filter only classes of interest
    print(Pcl['geocode_2'].value_counts())
    # regroup, geocode_2 from here on becomes binary!
    Pcl['geocode_2'] = np.where(Pcl['geocode_2'].str.contains(classes[k]),classes[k],'Others')
    print(Pcl['geocode_2'].value_counts())
    minc = min(Pcl['geocode_2'].value_counts() ) # skip if less than 20 objects 
    if minc< 20:
        print("minimum class less than 20")
        return (-1, -1) # -1 -1 if not calculated
    else:    
        print(f'total {len(df1)}, P_H1_classes: {len(Pcl)}, minimun class: {minc}')       
        # bootstrap and get averaged accuracy
        avepre = np.zeros(1) # store all the xgb+tree precisions in each CV
        averec = np.zeros(1)
        for seeds in range(seednum):
            np.random.seed(seeds)
            #1. categorise the variable "area", the variable "area" is kept in the data frame, strictly it can be removed.  
            #2. use groupby to sample the same amount for each area category 
            # use 70% of area for training, get the index
            print (Pcl['area'].quantile([0, .25, .5, .75, 1]))
            Pcl['area_c'] = pd.cut(Pcl['area'],
                     bins=  Pcl['area'].quantile([0, .25, .5, .75, 1]).tolist()
                     labels=[ "q25", "q5", "q75", "Max"])
            
            print(Pcl["area_c"].value_counts())

            train_ind = Pcl.groupby('area_c').sample(n = int(min(Pcl["area_c"].value_counts())*0.7)).index 
            test_ind = Pcl[~Pcl.index.isin(train_ind)].index
            
            Pcl.loc [train_ind,"geocode_2" ].value_counts()
            X_train0 = Pcl.loc [train_ind ].drop(columns=["geocode_2","layer","OBJECTID","path", "area_c"])
            X_test0  = Pcl.loc [test_ind ].drop(columns=["geocode_2","layer","OBJECTID","path", "area_c"])
            
            Y_train0 = Pcl.filter(regex='geocode_2').loc[train_ind] 
            Y_test0  = Pcl.filter(regex='geocode_2').loc[test_ind] 
            print("after sampling by area: for 2 classes,", X_train0.shape[0], X_test.shape[0])
            print(Pcl.loc [train_ind ]["geocode_2"].value_counts())
            # if my pandas is lower and i can't use the above function,
             
            # grouped = Pcl.drop(columns=["geocode_2","layer","OBJECTID","path",'area']).groupby('area_c')
            
            #def fun1(x):
            #    y = x.drop(columns=["area_c"]) 
            #    return( y.sample(n = int(minc/5*0.7)).index )
            #train_ind = grouped.apply(fun1) 
            #test_ind = Pcl[~Pcl.index.isin(train_ind)].index
            #neew to ungroup train_ind
            
            # test data
            #grouped2 = Pcl[['geocode_2',"area_c"]].groupby('area_c')
            #y = grouped2.apply(fun1)
            
            #####
            # after getting x, y train, we will use undersample to sample from each classes, p1a1 and others
            
            rus = RandomUnderSampler(random_state  = 1)
            X_train, Y_train = rus.fit_resample(X_train0, Y_train0)
            print("number of samples used for training:", X_train.shape[0]/2)
            #y2 = y2.reshape(-1, 1)
            #y2_rus, y_rus = rus.fit_resample(y2, y)
            #y2_rus= y2_rus.flatten()
           
            #len(train)+len(test)
            
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
           
             
            Y_trainnum = cl2idx(Y_train.values, class2idx).astype(int)
            Y_testnum = cl2idx(Y_test.values, class2idx).astype(int)
             
            np.unique(Y_trainnum)
            params = {'max_depth': 6, 'eta': 0.002, 
                      'objective':'binary:logistic', 'num_class': 1}
             
            clf = xgb.XGBModel(**params)

            clf.fit(X_train.values, Y_trainnum,
            eval_set=[(X_train.values, Y_trainnum), (X_test.values, Y_testnum)],
            eval_metric='logloss',
            verbose=True)
            
            #for testing
            #clf = DecisionTreeClassifier(min_samples_split= 30, max_depth= 4, min_samples_leaf=20, random_state=1)

            yhat = clf.predict(X_test)
                     
                    # threshold 0.5, probability higher than 0.5 -> positive. 
            yhat_labels = yhat>threshold
            yhat_labels = yhat_labels.astype(int)
            
 
            #TP
            TP = ((Y_testnum == 1) & (yhat_labels == 1)).astype(float) * X_test["area"]
            #FP
            FP = ((Y_testnum == 0) & (yhat_labels == 1)).astype(float) * X_test["area"]
            #TN
            TN = ((Y_testnum == 0) & (yhat_labels == 0)).astype(float) * X_test["area"]
            #FN
            FN =((Y_testnum == 1) & (yhat_labels == 0)).astype(float) * X_test["area"]
            precision = np.sum(TP)/np.sum(TP+FP) 
            recall = np.sum(TP)/np.sum(TP+TN) 
            

            averec = np.append(averec, recall) #store all of them
            avepre = np.append(avepre, precision)

        recall = averec.sum()/seednum #get the mean but exclude the first one (0)
        precision = avepre.sum()/seednum
        print(averec, recall)
        if resultdir is not None:
            Y_testnum =  Y_testnum.astype(int)
            plt.rcParams.update({'font.size': 8})
            ax = xgb.plot_importance(model, grid=False, importance_type='gain', title='Feature importance')
            ax.set_title(f'xgboost importance {layers[j]} {classes[k]}')
            fname = f"{resultdir}/P_{layers[j]}_{classes[k]}_imp"
            plt.savefig(fname, dpi=1200)
        return (recall, precision)
 
#filedir = "/Users/menglu/Documents/GitHub/waaden/OBIA_RF/results" # for confusion matrices
#resultdir = "/Users/menglu/Volumes/Meng_Mac/obia/results/" # for var imp plots

recall_xgbtree, precision_xgbtree = run(k = 0,j = 3, filename = filename,resultdir=resultdir, seednum=20, threshold = 0.5, graphdir =f'{treedir}' )
print(recall_xgbtree, precision_xgbtree