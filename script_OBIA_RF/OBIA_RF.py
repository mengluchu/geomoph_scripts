import os 
import geopandas as gp
import pandas as pd
import sklearn 
from sklearn.ensemble import RandomForestClassifier
import xgboost
from sklearn.metrics import confusion_matrix

def cl2idx(inputarr, dict_label):
	y_label = np.zeros((inputarr.shape[0] ))
	for i in range(inputarr.shape[0]):
			y_label[i] = dic_label[inputarr[i, 0]]
	return y_label
	
	
os.chdir("/data/lu01/Objects_with_properties")
alld = gp.read_file("objects.shp")
allclass = gp.read_file("/data/lu01/2016_GMK/e_GMK_Westerschelde2016.shp")
joind = gd.sjoin(alld, allclass, how="inner", op='intersects')


 
df1 = pd.DataFrame(joind.drop(columns='geometry'))
df1 = df1.replace([np.inf, -np.inf], np.nan).dropna()
 
df_covar = df1.filter(regex='density|Dif_|EF|GLCM|LW|max_SI|mean_|mn_|num|obj|SI_|std|tot')
df_covar[df_covar>1e5] =0 
 
train_size = int(len(df1)*0.8)
X_train =df_covar [:train_size]
X_test  = df_covar  [train_size:]
Y_train  =df1.filter(regex='OMS_GEOCOD')[:train_size]
Y_test  =df1.filter(regex='OMS_GEOCOD')[train_size:]
Y_train= Y_train.values
Y_test = Y_test.values
 

label_all =  df1.OMS_GEOCOD.unique()
i = 0
idx2class = {}
class2idx = {}
for tp in label_all:
	idx2class[i] = tp
	class2idx[tp] = i 
	i+= 1

 #string.ascii_lowercase
 
 
	
Y_trainnum = cl2idx(Y_train, class2idx)
Y_testnum = cl2idx(Y_test, class2idx)
 
clf = RandomForestClassifier(max_depth=5, random_state=0,criterion='gini',n_estimators = 1000)
clf.fit(X_train, Y_trainnum)
clf.score(X_test, Y_testnum)
decision_path(X_train)[source]


Y = OneHotEncoder(sparse=False).fit_transform(Ymc.reshape(-1, 1))


"""xgboost softmax regression"""
dtrain = xgb.DMatrix(X_train, label=Y_trainnum)
dtest = xgb.DMatrix(X_test, label=Y_testnum)
params = {'max_depth': 2, 'eta': 0.1, 'silent': 1,
          'objective': 'multi:softprob', 'num_class': len(np.unique(Y_train))}
# Fit
model = xgb.train(params, dtrain, 100)
# Evalute
yhat = model.predict(dtest)
yhat_labels = np.argmax(yhat, axis=1)
confusion_matrix(Y_testnum, yhat_labels)
 
 
 