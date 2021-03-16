# %% [markdown]
# import packages

# %% [code]
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


# %% [code]
!pip install segmentation-models-pytorch==0.1.0
!pip install keras-unet
!pip install pytorch-lighting
 

# %% [code]
import segmentation_models_pytorch as smp
from keras_unet.utils import plot_imgs

import pytorch_lightning as pl

# %% [code]
size = 128
bs = 16 #batchsize
classes = [7]
num_class = len(classes)
EPOCH = 300
reduce = 8

# %% [markdown]
# Used to crop the imput images

# %% [code]
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

# %% [markdown]
# load the different tiles into 1 variable

# %% [code]
#stack all files into 1 variable
def load_raster(path,data,tile1,tile2,reduce):
   tiles_in=[tile1,tile2]
   files = []
   for tile in tiles_in:
       for file in glob.glob(path+data+'*{}.tif'.format(tile)):
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


# %% [code]
def load_raster2(path,data,tile1,tile2,tile3,tile4,reduce):
   tiles_in=[tile1,tile2,tile3,tile4]
   files = []
   for tile in tiles_in:
       print(tile)
       for file in glob.glob(path+data+'*{}.tif'.format(tile)):
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

# %% [markdown]
# 2 tiles

# %% [markdown]
# x_train0= load_data("../input/guided-research/Training/","DEM","Wadden","12_7","13_8",reduce)
# x_val0 = load_data("../input/guided-research/Training/","DEM","Wadden","12_8","14_7",reduce)
# y_train0= load_raster("../input/guided-research/Training/","class","12_7","13_8", reduce )
# y_val0=load_raster("../input/guided-research/Training/","class","12_8","14_7",reduce )

# %% [markdown]
# 4 tiles

# %% [code]
x_train0= load_data2("../input/guided-research/Training/","DEM","Wadden","12_7","12_8","13_8","14_7",reduce)
x_val0 = load_data("../input/guided-research/Validation/","DEM","Wadden","15_7","15_8",reduce)
y_train0= load_raster2("../input/guided-research/Training/","class","12_7","12_8","13_8","14_7", reduce )
y_val0=load_raster("../input/guided-research/Validation/","class","15_7","15_8",reduce )

# %% [code]
y_val0.shape

# %% [code]
np.sum(y_val0[0,]==7)

# %% [code]
print(x_train0.shape,x_val0.shape,y_train0.shape,y_val0.shape)

# %% [code]
print(np.unique(y_train0),np.unique(y_train0))

# %% [markdown]
# Make class maps binary, 0 for other class 1 for 5

# %% [code]
y_train0 = (y_train0==7).astype(int)
y_val0 = (y_val0==7).astype(int)
print(np.unique(y_val0),np.unique(y_train0))

# %% [code]
np.sum(y_train0)

# %% [code]
for i in range (2):
    plt.imshow(y_val0[i,])
    plt.colorbar()
    plt.show()

# %% [code]
for i in range (4):
    plt.imshow(y_train0[i,])
    plt.colorbar()
    plt.show()

# %% [code]
def process(x_train, x_val, y_train, y_val,  size, bslice=True, cl2dim=True, Inf2zero=True):
    if bslice :
        x_train  = batchslice(x_train, size,x_train[0].shape[1],size, x_train.shape[0])
        x_val = batchslice(x_val,size, x_val[0].shape[1], size, x_val.shape[0])
        y_train = batchslice(y_train,size,y_train[0].shape[1],size,y_train.shape[0]).squeeze()
        y_val = batchslice(y_val,size,y_val[0].shape[1],size,y_val.shape[0]).squeeze()
        print(f"batch slice to {size}")
       
    if  cl2dim :    
        y_train = class2dim(y_train, classes)
        y_val = class2dim(y_val, classes)

        y_train=  np.moveaxis(y_train, -1, 1)
        y_val = np.moveaxis(y_val, -1, 1)
        print("classes are converted to channels")
        
    if Inf2zero : 
        x_train[x_train > 1e308] = 0 
        #np.nan
        x_val [x_val > 1e308] =0 
        y_train[y_train >1e308] = 0
        y_val[y_val> 1e308] = 0
        print("inf are converted to 0")
    return(x_train, x_val, y_train, y_val)

def myloader(trainX, trainY, valX,valY):

    #train = TensorDataset(torch.Tensor(x_train[:,1:4,:,:]), torch.Tensor(y_train )) # create your datset
    train = TensorDataset(torch.Tensor(x_train[:,:,:,:]), torch.Tensor(y_train )) # create your datset
    train  = DataLoader(train, batch_size=bs) # create your dataloader
    
    #vali= TensorDataset(torch.Tensor(x_val[:,1:4,:,:]),torch.Tensor(y_val   )) # create your datset
    vali= TensorDataset(torch.Tensor(x_val[:,:,:,:]),torch.Tensor(y_val   )) # create your datset
    
    vali = DataLoader(vali, batch_size=bs) # create your dataloader
    return train , vali 

# %% [code]
x_train, x_val, y_train, y_val = process(x_train0, x_val0, y_train0, y_val0, size =size, bslice=True, cl2dim=False, Inf2zero=True)

# %% [code]
print(x_train.shape, x_val.shape, y_train.shape, y_val.shape)

# %% [code]
train_loader, valid_loader = myloader(x_train, y_train, x_val, y_val)

# %% [code]
GAMMA = 2
ALPHA = 0.8 # emphasize FP
BETA = 0.2 # more emphasize on FN

# combo loss
cl_ALPHA = 0.5 # < 0.5 penalises FP more, > 0.5 penalises FN more
cl_BETA = 0.5
CE_RATIO = 0.5 #weighted contribution of modified CE loss compared to Dice loss
e=1e-07

# %% [code]
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs)       
        
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
        inputs = torch.where(inputs < 0.5,torch.tensor(0.).cuda(), inputs)
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

# %% [code]
class ComboLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(ComboLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=cl_ALPHA, beta=cl_BETA):
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        intersection = (inputs * targets).sum()    
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        
        inputs = torch.clamp(inputs, e, 1.0 - e)       
        out = - (cl_ALPHA * ((targets * torch.log(inputs)) + ((1 - cl_ALPHA) * (1.0 - targets) * torch.log(1.0 - inputs))))
        weighted_ce = out.mean(-1)
        combo = (CE_RATIO * weighted_ce) - ((1 - CE_RATIO) * dice)
        
        return combo

# %% [code]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

# %% [code]
class myCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(ComboLoss, self).__init__()

    def forward(self, outputs, labels, bs):


        batch_size = outputs.size()[0]            # batch_size
        outputs = F.log_softmax(outputs, dim=1)   # compute the log of softmax values
        outputs = outputs[range(batch_size), labels] # pick the values corresponding to the labels
        return -torch.sum(outputs)/bs

# %% [code]
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

# %% [code]
class CNNmodel(pl.LightningModule):
    def __init__(self):
        super(CNNmodel,self).__init__()
        self.batch_size=bs
        self.learning_rate = 2e-4
        self.net = smp.Unet(classes=num_class, in_channels=4, activation = 'sigmoid')
        self.label_type = torch.float32 if num_class  == 1 else torch.long
        self.accuracy = pl.metrics.Accuracy()
    def forward(self,x):
        return self.net(x)
    
    def training_step(self,train_batch,batch_nb):
        x,y = train_batch
        y = y.float()
        y_hat = self.net(x)
        #loss1= IoULoss()
        loss1=DiceBCELoss()
        loss = loss1(y_hat, y)
        
        self.log('train_acc_step', loss)
        return loss
    
    def validation_step(self,val_batch,batch_nb):
        x,y = val_batch
        y = y.float()
        y_hat = self.net(x)
        #loss1 = IoULoss()
        loss1=DiceBCELoss()
        val_loss = loss1(y_hat, y)
        
        
        result = pl.EvalResult(checkpoint_on=val_loss)
        result.log('val_loss', val_loss, prog_bar=True, on_step=False, on_epoch=True)
        return result
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 20, 40], gamma=0.3)
        return [optimizer],[scheduler]
    
    def train_dataloader(self):
        return train_loader
    
    def valid_dataloader(self):
        return valid_loader

# %% [code]
from pytorch_lightning.callbacks import ModelCheckpoint

# %% [code]
model = CNNmodel()
#logdir_lightn = "segmentation_notebook_light"
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    filepath='sample-{epoch:02d}',
    prefix='')
trainer = pl.Trainer(gpus=1,max_epochs = EPOCH,checkpoint_callback=checkpoint_callback)
modelt = trainer.fit(model,train_loader,valid_loader)

# %% [code]
print(checkpoint_callback.best_model_path,checkpoint_callback.best_model_score)
from IPython.display import FileLink
FileLink(checkpoint_callback.best_model_path[16:])

# %% [code]
import IPython.display as ipd
audio_path="https://www.soundjay.com/button/beep-01a.wav"

ipd.Audio(audio_path, autoplay=True)

# %% [markdown]
# after download changes the name to s_(#traingin_tiles)_(size)

# %% [code]

new_model = CNNmodel.load_from_checkpoint(checkpoint_path=checkpoint_callback.best_model_path,
                                         maplocation="gpu")

# %% [code]
 
test_loader = valid_loader
#new_model =new_model.cuda()
trainer=pl.Trainer(gpus=1)
result_model = trainer.test(new_model,test_dataloaders=test_loader)
#%%21]:

# %% [code]
x_test0=x_val0
x_test = x_val
y_test0=y_val0
y_test = y_val
import pandas as pd
nr_tiles=x_val0.shape[0]

numimag = np.sqrt(x_test.shape[0]/nr_tiles).astype(int)
num = (numimag*numimag+1).astype(int)
df_tot=pd.DataFrame(data=0,index=range(3),columns=range(3))
df_list=[]

classes = [5]
big = np.zeros((numimag*size, numimag*size))
result = np.zeros((1,size, size))
result2 = np.zeros((1,size, size))

for plot in range(1,nr_tiles+1):
    with torch.no_grad():
        for data in test_loader:
            images, labels = data 
            
            output1 = new_model(images.cuda())
            
            p1 = (output1> .5).cpu().numpy().astype(int).squeeze()
            result = np.concatenate((result, p1),axis=0)
            p2,_ = torch.max(output1.cpu(),  1)
            result2 = np.concatenate((result2, p2.numpy()),axis=0)
    print(plot)
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
    plt.imshow(y_test0[plot-1,],cmap=cmap,norm=norm)
    plt.colorbar(ticks=ticks,boundaries=bounds,fraction=0.045)
    plt.title(f'validation')
    
    plt.subplot(1,4,4)
    try1 = np.moveaxis(x_test0[plot-1,1:,],0,-1)
    try1= try1.astype(int)
    plt.imshow(try1,cmap=plt.get_cmap("PRGn"))
    #plt.colorbar()
    plt.title(f'false color')
# =============================================================================
#    plt.savefig(os.path.join(savepath,"plot{}.tif".format(plot)))
# =============================================================================
    plt.show()    

# %% [code]
 
    
    conf_matrix = pl.metrics.functional.classification.confusion_matrix(torch.from_numpy(y_label),torch.from_numpy(big),False)
    conf = conf_matrix.numpy()
    dx = pd.DataFrame(data=conf,dtype=int)
    df=pd.DataFrame(columns=range(3),index=range(3),dtype=int).fillna(0)
    df = df.add(dx).fillna(0)
    df=df.iloc[:3,:3]
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
df_tot=df_tot.fillna(0)
diag = np.diag(df_tot)
prod = (diag/df_tot.sum(1))*100
user=(diag/df_tot.sum(0))*100
overall =np.diag(df_tot).sum()/df_tot.sum().sum()
df_tot['producer_acc'] = round(prod,2)
df_tot.loc['user_acc'] = round(user,2)
df_tot.loc['user_acc','producer_acc'] = round(overall,3)
df_tot=df_tot.fillna(0)
print(df_tot.loc['user_acc','producer_acc'])
df_tot.to_csv(r'{}\tot.csv'.format(savepath), header=True, index=True, sep=',')
