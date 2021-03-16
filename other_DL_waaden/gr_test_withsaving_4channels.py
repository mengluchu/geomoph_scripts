# =============================================================================
# gr_test-Withsaving but with 4 classes
# =============================================================================
import torch
import torchvision
import torchvision.transforms as transforms
from osgeo import gdal_array
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from torch import Tensor, einsum
from torch import nn
from torch.nn import functional as F
from scipy.ndimage import distance_transform_edt as distance
from scipy.spatial.distance import directed_hausdorff
from typing import Any, Callable, Iterable, List, Set, Tuple, TypeVar, Union
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
import matplotlib
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import pytorch_lightning as pl
#%%
size = 128
ep_nr = '16_4'
training_tiles = 8 #2 or 4 or 8
bs = 16 #batchsize
classes = [ 1,  2,   4,  5,  6,  7,  8,  9, 10]
num_class = len(classes)
EPOCH = 300
reduce = 8
savepath = os.path.join(r'C:\Data\uni\uni\Master\Guided-Research\results',"{}_{}_{}_4".format(training_tiles,size,EPOCH))
if not os.path.exists(savepath): os.mkdir(savepath)
#%%19]:
def slice (arr, size, inputsize,stride):
    result = []
    if stride is None:
        stride = size
    for i in range(0, (inputsize-size)+1, stride):
        for j in range(0, (inputsize-size)+1, stride):
        
            if arr.ndim == 3:
                s = arr[:,i:(i+size),j:(j+size), ]
            else:
                s = arr[i:(i+size),j:(j+size), ]
            result.append(s)
            #print(i,"",j)
    result = np.array(result)
    return result

def batchslice (arr, size, inputsize, stride, num_img):
    result = []
    for i in range(0, num_img):
        s= slice(arr[i,], size, inputsize, stride )
        result.append(s )
    result = np.array(result)
    result = result.reshape(result.shape[0]*result.shape[1], result.shape[2], result.shape[3], -1)
    return result

def class2dim (mask, CLASSES):
    
        masks = [(mask == v) for v in CLASSES ]
        mask = np.stack(masks, axis=-1).astype('float')    
        return mask
#stack all files into 1 variable
def load_raster(path,data,tile1,tile2,reduce):
   tiles_in=[tile1,tile2]
   files = []
   for tile in tiles_in:
       for file in glob.glob(data+'*{}.tif'.format(tile)):
           
           file1 = gdal_array.LoadFile(file)
           #only use 50% of the points to reduce memory
           if np.ndim(file1)==3:
               file1=file1[:,::reduce,::reduce]
           else:
            
               file1=file1[::reduce,::reduce]
           files.append(file1)
   stacked = np.array(files)
   return stacked

def load_data(path,data1,data2,tile1,tile2, reduce):
    part1 = load_raster(path,data1,tile1,tile2, reduce )
    part2 = load_raster(path,data2,tile1,tile2,reduce )
    
    if np.ndim(part1)< np.ndim(part2):#check if dimmensions are equal
       part1 = np.expand_dims(part1,axis=1)
    elif np.ndim(part1)> np.ndim(part2):
       part2 = np.expand_dims(part2,axis=1)
    print(part1.shape,part2.shape)
    total = np.concatenate((part1,part2),axis=1)
    return total
#stack all files into 1 variable
def load_raster2(path,data,tile1,tile2,tile3,tile4,reduce):
   tiles_in=[tile1,tile2,tile3,tile4]
   files = []
   for tile in tiles_in:
       print(tile)
       for file in glob.glob(data+'*{}.tif'.format(tile)):
           file1 = gdal_array.LoadFile(file)
           #only use 50% of the points to reduce memory
           if np.ndim(file1)==3:
               file1=file1[:,::reduce,::reduce]
           else:
               file1=file1[::reduce,::reduce]
           files.append(file1)
           print(file1.shape)
   stacked = np.array(files)
   return stacked

def load_data2(path,data1,data2,tile1,tile2,tile3,tile4, reduce):
    part1 = load_raster2(path,data1,tile1,tile2,tile3,tile4,reduce )
    part2 = load_raster2(path,data2,tile1,tile2,tile3,tile4,reduce )
    
    if np.ndim(part1)< np.ndim(part2):#check if dimmensions are equal
       part1 = np.expand_dims(part1,axis=1)
    elif np.ndim(part1)> np.ndim(part2):
       part2 = np.expand_dims(part2,axis=1)
    print(part1.shape,part2.shape)
    total = np.concatenate((part1,part2),axis=1)
    return total
#%%21]:Org test data
# dem= r'C:\Data\uni\uni\Master\Guided-Research\Data_GR_Addink\EI_ZEEGAT\dem_tiles\resample1600\tile_'
# wadden =r'C:\Data\uni\uni\Master\Guided-Research\Data_GR_Addink\EI_ZEEGAT\tif_tiles\Wadden_2017_tile_'
# class_path=r'C:\Data\uni\uni\Master\Guided-Research\Data_GR_Addink\class_main_tiles\class_main_'
# x_test0= load_data("Test/",dem,wadden,"14_6","14_8",reduce)
# y_test0=load_raster("Test/",class_path,"14_6","14_8",reduce )
# #%%
# dem= r'C:\Data\uni\uni\Master\Guided-Research\Data_GR_Addink\EI_ZEEGAT\dem_tiles\resample1600\tile_'
# wadden =r'C:\Data\uni\uni\Master\Guided-Research\Data_GR_Addink\EI_ZEEGAT\tif_tiles\Wadden_2017_tile_'
# class_path=r'C:\Data\uni\uni\Master\Guided-Research\Data_GR_Addink\class_main_tiles\class_main_'
# x_test0= load_data("..",dem,wadden,"14_9","15_9",reduce)
# y_test0=load_raster("..",class_path,"14_9","15_9",reduce )
#%%
dem= r'C:\Data\uni\uni\Master\Guided-Research\Data_GR_Addink\EI_ZEEGAT\dem_tiles\resample1600\tile_'
wadden =r'C:\Data\uni\uni\Master\Guided-Research\Data_GR_Addink\EI_ZEEGAT\tif_tiles\Wadden_2017_tile_'
class_path=r'C:\Data\uni\uni\Master\Guided-Research\Data_GR_Addink\class_main_tiles\class_main_'
x_test0= load_data2("..",dem,wadden,"14_6","14_8","14_9","15_9",reduce)
y_test0=load_raster2("..",class_path,"14_6","14_8","14_9","15_9",reduce )
#%%
print(x_test0.shape,y_test0.shape)
def process(x_test,y_test , size, bslice=True, cl2dim=True, Inf2zero=True):
    if bslice :
        x_test  = batchslice(x_test, size,x_test[0].shape[1],size, x_test.shape[0])
        y_test = batchslice(y_test,size,y_test[0].shape[1],size,y_test.shape[0]).squeeze()
        print(f"batch slice to {size}")
       
    if  cl2dim :    
        y_test = class2dim(y_test, classes)

        y_test=  np.moveaxis(y_test, -1, 1)
        print("classes are converted to channels")
        
    if Inf2zero : 
        x_test[x_test > 1e308] = 0 
        #np.nan
        y_test[y_test >1e308] = 0
        print("inf are converted to 0")
    return(x_test, y_test)

def myloader(testX, testY):

    test = TensorDataset(torch.Tensor(x_test[:,:,:,:]), torch.Tensor(y_test )) # create your datset
    test  = DataLoader(test, batch_size=bs) # create your dataloader
    
    return test
x_test, y_test = process(x_test0, y_test0, size =size, bslice=False, cl2dim=True, Inf2zero=True)
x_test, y_test = process(x_test0, y_test0, size =size, bslice=True, cl2dim=True, Inf2zero=True)
print(x_test.shape, y_test.shape)
test_loader = myloader(x_test, y_test)
GAMMA = 2
ALPHA = 0.8 # emphasize FP
BETA = 0.2 # more emphasize on FN

# combo loss
cl_ALPHA = 0.5 # < 0.5 penalises FP more, > 0.5 penalises FN more
cl_BETA = 0.5
CE_RATIO = 0.5 #weighted contribution of modified CE loss compared to Dice loss
e=1e-07
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE


     
class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True, **kwargs):
        super(IoULoss, self).__init__(**kwargs)

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs)       
        #print(inputs.shape)
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        #inputs = (inputs>0.5).float()
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()  
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth) 
                
        return 1 - IoU
    
class TverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=ALPHA, beta=BETA):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
       
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        
        return 1 - Tversky
 

class FocalTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=ALPHA, beta=BETA, gamma=GAMMA):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
        
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        FocalTversky = (1 - Tversky)**gamma
                       
        return FocalTversky

class myLoss(torch.nn.Module):

    def __init__(self, pos_weight=1):
      super().__init__()
      self.pos_weight = pos_weight

    def forward(self, input, target):
      epsilon = 10 ** -44
      input = input.sigmoid().clamp(epsilon, 1 - epsilon)

      my_bce_loss = -1 * (self.pos_weight * target * torch.log(input)
                          + (1 - target) * torch.log(1 - input))
      add_loss = (target - 0.5) ** 2 * 4
      mean_loss = (my_bce_loss * add_loss).mean()
      return mean_loss
    
 

class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device
class CNNmodel(pl.LightningModule):
    def __init__(self):
        super(CNNmodel,self).__init__()
        self.batch_size=bs
        self.learning_rate = 2e-4
        self.net = smp.Unet(classes=num_class, in_channels=4, activation = 'softmax')
        self.label_type = torch.float32 if num_class  == 1 else torch.long
    
    def forward(self,x):
        return self.net(x)
    
    def training_step(self,train_batch,batch_nb):
        x,y = train_batch
        y = y.float()
        y_hat = self(x)
        loss1=IoULoss()
        loss = loss1(y_hat, y)
        return{'loss':loss}
    
    def validation_step(self,val_batch,batch_nb):
        x,y = val_batch
        y = y.long()
        y_hat = self(x)
        loss1=IoULoss()
        val_loss = loss1(y_hat, y)
        return val_loss
    def test_step(self,test_batch,batch_nb):
        x,y = test_batch
        y = y.long()
        y_hat = self(x)
        loss=IoULoss()
        loss = loss(y_hat, y)
        self.log('test_loss',loss)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 20, 40], gamma=0.3)
        return [optimizer],[scheduler]
    
    def train_dataloader(self):
        return train_loader
    
    def valid_dataloader(self):
        return valid_loader
    def test_dataloader(self):
        return test_loader

# print(os.listdir(R'C:\Data\uni\uni\Master\Guided-Research\model_param'))
#%%18]:
model=r"C:\Data\uni\uni\Master\Guided-Research\model_param\s_{}_{}-epoch={}.ckpt".format(training_tiles,size,ep_nr)
new_model = CNNmodel.load_from_checkpoint(checkpoint_path=model,
                                         maplocation="gpu")
#new_model =new_model.cuda()
trainer=pl.Trainer(gpus=1)
result_model = trainer.test(new_model,test_dataloaders=test_loader)
#%%21]:
import pandas as pd
nr_tiles=x_test0.shape[0]
numimag = np.sqrt(x_test.shape[0]/nr_tiles).astype(int)
num = (numimag*numimag+1).astype(int)
df_tot=pd.DataFrame(data=0,index=range(9),columns=range(9))
df_list=[]

classes = [1,2,4,5,6,7,8,9,10]
big = np.zeros((numimag*size, numimag*size))
result = np.zeros((1,size, size))
result2 = np.zeros((1,size, size))

for plot in range(1,nr_tiles+1):
    with torch.no_grad():
        for data in test_loader:
            images, labels = data 
            output1 = new_model(images.cuda())
            p1 = torch.argmax(output1.cpu(), dim = 1).cpu().numpy()
            result = np.concatenate((result, p1),axis=0)
            p2,_ = torch.max(output1.cpu(),  1)
            result2 = np.concatenate((result2, p2.numpy()),axis=0)
    print(plot)
    label_all = range(10) 
    ori_val = classes # random numpy numbers
    enum_val = [x for n, x in enumerate(ori_val) if x not in ori_val[:n]]
    sort_val = np.sort(enum_val)
    dic_label = {}
    for ind, v in enumerate(sort_val):
        dic_label[v] = label_all[ind]
        dic_label[3]=9
    dic_label[15]=9
    print(dic_label) 
    
    y_label = np.zeros((big.shape[0],big.shape[1]))
    for i in range(big.shape[0]):
        for j in range(big.shape[1]):
            y_label[i,j] = dic_label[y_test0[(plot-1),i,j]]        
    if plot == 1:
        result_plot = result[1:num,:,:] 
        result2_plot = result2[1:num,:,:]      
    elif plot == 2:
        result_plot = result[num:(num*2),:,:] 
        result2_plot = result2[num:(num*2),:,:]   
    elif plot == 3:
        result_plot = result[(num*2)-1:(num*3),:,:] 
        result2_plot = result2[(num*2)-1:(num*3):,:,:]   
    elif plot == 4:
        result_plot = result[(num*3)-2:,:,:] 
        result2_plot = result[(num*3)-2:,:,:] 
    
    result_plot= np.moveaxis(result_plot, 0, -1)     
    result2_plot= np.moveaxis(result2_plot, 0, -1)
    big = np.zeros((numimag*size, numimag*size))
    big2 = np.zeros((numimag*size, numimag*size))
     
    for j in range(numimag):    
            for i in range(numimag):
                big[j*size: (j+1)*size,i*size: (i+1)*size]= result_plot[:,:,i+j*numimag]
                big2[j*size: (j+1)*size,i*size: (i+1)*size]= result2_plot[:,:,i+j*numimag]
                
    cmap = matplotlib.colors.ListedColormap(matplotlib.cm.get_cmap("Set3").colors[:9])
    bounds= np.linspace(-.5,8.5,10)
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    ticks=[0,1,2,3,4,5,6,7,8]
    fig = plt.figure(figsize=(18,18))
    plt.subplot(1,4,1)
    plt.title('prediction:{}'.format(result_model))
    plt.imshow(big,cmap=cmap,norm=norm)
    plt.colorbar(ticks=ticks,boundaries=bounds,fraction=0.045)
    
    
    plt.subplot(1,4,2)
    plt.title('probability')
    plt.imshow(big2,cmap="PRGn")
    plt.colorbar(fraction=0.045)
    
    
    plt.subplot(1,4,3)
    plt.imshow(y_label,cmap=cmap,norm=norm)
    plt.colorbar(ticks=ticks,boundaries=bounds,fraction=0.045)
    plt.title(f'validation')
    
    plt.subplot(1,4,4)
    try1 = np.moveaxis(x_test0[plot-1,1:,],0,-1)
    try1= try1.astype(int)
    plt.imshow(try1,cmap=plt.get_cmap("PRGn"))
    #plt.colorbar()
    plt.title(f'false color')
# =============================================================================
    plt.savefig(os.path.join(savepath,"plot{}.tif".format(plot)))
# =============================================================================
    plt.show()    
    
    conf_matrix = pl.metrics.functional.classification.confusion_matrix(torch.from_numpy(y_label),torch.from_numpy(big),False)
    conf = conf_matrix.numpy()
    dx = pd.DataFrame(data=conf,dtype=int)
    df=pd.DataFrame(columns=range(9),index=range(9),dtype=int).fillna(0)
    df = df.add(dx).fillna(0)
    df=df.iloc[:9,:9]
    df_tot=df_tot+df
    
    diag = np.diag(df)
    prod = (diag/df.sum(1))*100
    user=(diag/df.sum(0))*100
    overall =np.diag(df).sum()/df.sum().sum()
    df['producer_acc'] = round(prod,2)
    df.loc['user_acc'] = round(user,2)
    df.loc['user_acc','producer_acc'] = round(overall,3)
    print(df.loc['user_acc','producer_acc'])
    df=df.fillna(0)
    df_list.append(df)
    df.to_csv(r'{}\tile{}.csv'.format(savepath,plot), header=True, index=True, sep=',')
#calc total values
# df_tot =df_tot.iloc[:9,:9]
df_tot=df_tot.fillna(0)
diag = np.diag(df_tot)
prod = (diag/df_tot.sum(1))*100
user=(diag/df_tot.sum(0))*100
overall =np.diag(df_tot).sum()/df_tot.sum().sum()
df_tot['producer_acc'] = round(prod,2)
df_tot.loc['user_acc'] = round(user,2)
df_tot.loc['user_acc','producer_acc'] = round(overall,3)
df_tot=df_tot.fillna(0)
np.savetxt(r'c:\data\np.txt', df.values, fmt='%d')
print(df_tot.loc['user_acc','producer_acc'])
df_tot.to_csv(r'{}\tot.csv'.format(savepath), header=True, index=True, sep=',')
