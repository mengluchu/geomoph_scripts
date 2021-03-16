import os
import geopandas as gp
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier

og = gp.read_file("/data/OBIA_GMK.gdb")
allclass = og[["GEOCODE2","geometry"]]

filedir ="/data/lu01/Objects_with_properties/"
alld = gp.read_file(filedir+"objects.shp")
#allclass = gp.read_file("/data/lu01/2016_GMK/e_GMK_Westerschelde2016.shp")
#joind = gd.sjoin(alld, allclass, how="inner", op='intersects')
joind = gp.sjoin(alld, allclass, how="inner", op='within')
joind.to_file("/data/lu01/obia/joined/joind_intersects.gpkg")

# with R: st_join(x,y,join = st_intersects, left = TRUE, largest = True)
 
df1 = pd.DataFrame(joind.drop(columns='geometry'))
df1 = df1.replace([np.inf, -np.inf], np.nan).dropna()
 
df_covar = df1.filter (regex=‘density|Dif_|EF|GLCM|LW|max_SI|mean_|mn_|num|obj|SI_|std|tot’)
train_size = int(len(df1)*0.8)
X_train =df_covar [:train_size]
X_test  = df_covar  [train_size:]
Y_train  =df1.filter(regex='OMS_GEOCOD')[:train_size]
Y_test  =df1.filter(regex='OMS_GEOCOD')[train_size:]
Y_train= Y_train.values
Y_test = Y_test.values
 
 
clf = RandomForestClassifier(max_depth=5, random_state=0)
clf.fit(X_train, Y_train)

xg_reg = xgboost.XGBRegressor(objective = 'multi:softmax',booster = 'dart', learning_rate = 0.007, max_depth =6 , n_estimators = 1000,gamma =5, alpha =2) 

label_all =  df1.OMS_GEOCOD.unique()
i = 0
idx2class = {}
class2id = {}
for tp in label_all:
	idx2class[i] = tp
	class2idx[tp] = i 
	i+= 1

 #string.ascii_lowercase
 
ori_val = classes # random numpy numbers

enum_val = [x for n, x in enumerate(ori_val) if x not in ori_val[:n]]

sort_val = np.sort(enum_val)
dic_label = {}
for ind, v in enumerate(sort_val):
    dic_label[v] = label_all[ind]
print(dic_label) 

y_label = np.zeros((big.shape[0],big.shape[1]))
for i in range(big.shape[0]):
    for j in range(big.shape[1]):
        y_label[i,j] = dic_label[y_val0[0,i,j]]

xg_reg.fit(X_train ,Y_train) # predictor at the station and station measurements
y_hatxgb = xg_reg.predict(X_test) # 1 degree tile
 
