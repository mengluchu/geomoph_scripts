# =============================================================================
# code for figure 12
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
import pandas as pd
from mpl_toolkits.axes_grid1 import ImageGrid
import pickle
import matplotlib.gridspec as gridspec
#%%
reduce = 8
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
def load_raster3(path,data,tile1,tile2,tile3,tile4,reduce):
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

def load_data3(path,data1,data2,data3,tile1,tile2,tile3,tile4,reduce):
    part1 = load_raster3(path,data1,tile1,tile2,tile3,tile4,reduce)
    part2 = load_raster3(path,data2,tile1,tile2,tile3,tile4,reduce)
    part3 = load_raster3(path,data3,tile1,tile2,tile3,tile4,reduce)
    
    if np.ndim(part1)< np.ndim(part2):#check if dimmensions are equal
        part1 = np.expand_dims(part1,axis=1)
    elif np.ndim(part1)> np.ndim(part2):
        part2 = np.expand_dims(part2,axis=1)
    if np.ndim(part1)< np.ndim(part3):#check if dimmensions are equal
        part1 = np.expand_dims(part1,axis=1)
    elif np.ndim(part1)> np.ndim(part3):
        part3 = np.expand_dims(part3,axis=1)
    print(part1.shape,part2.shape,part3.shape)
    total = np.concatenate((part1,part2,part3),axis=1)
    return total
dem= r'C:\Data\uni\uni\Master\Guided-Research\Data_GR_Addink\EI_ZEEGAT\dem_tiles\resample1600\tile_'
wadden =r'C:\Data\uni\uni\Master\Guided-Research\Data_GR_Addink\EI_ZEEGAT\tif_tiles\Wadden_2017_tile_'
slope = r'C:\Data\uni\uni\Master\Guided-Research\Data_GR_Addink\EI_ZEEGAT\slope\slope_'
class_path=r'C:\Data\uni\uni\Master\Guided-Research\Data_GR_Addink\class_main_tiles\class_main_'
x_test0= load_data3("..",dem,wadden,slope,"14_6","14_8","14_9","15_9",reduce)
y_test0=load_raster2("..",class_path,"14_6","14_8","14_9","15_9",reduce )
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
        remove = np.unique(x_test[0,4,:,:])
        x_test[np.isinf(x_test)] = 0
        x_test[np.isnan(x_test)] = 0
        x_test[x_test == remove[0]] =0
        
        y_test[np.isinf(y_test)] = 0
        y_test[np.isnan(y_test)] = 0
    return(x_test, y_test)

def myloader(testX, testY,nr_channels):
    if nr_channels == 3:
        test = TensorDataset(torch.Tensor(x_test[:,1:4,:,:]), torch.Tensor(y_test )) # create your datset
    else:
        test = TensorDataset(torch.Tensor(x_test[:,:nr_channels,:,:]), torch.Tensor(y_test )) # create your datset
    test  = DataLoader(test, batch_size=bs) # create your dataloader
    return test
#%%
def ndvi(input1):
    im =np.zeros(shape=(0,2000,2000))
    for i in range(input1.shape[0]):
        nir = input1[i,1,:,:]
        vis = input1[i,2,:,:]
        ndvi = (nir-vis)/(nir+vis)
        ndvi=np.expand_dims(ndvi,axis=0)
        im = np.concatenate([im,ndvi],axis=0)
    im = np.expand_dims(im,axis=1)
    outp = np.concatenate([input1,im],axis=1)
    return outp
x_test0=ndvi(x_test0)
def brightness(input1):
    im =np.zeros(shape=(0,2000,2000))
    for i in range(input1.shape[0]):
        nir = input1[i,1,:,:]
        red = input1[i,2,:,:]
        green = input1[i,3,:,:]
        brightness = (nir+red+green)/3
        brightness=np.expand_dims(brightness,axis=0)
        im = np.concatenate([im,brightness],axis=0)
    im = np.expand_dims(im,axis=1)
    outp = np.concatenate([input1,im],axis=1)
    return outp
x_test0=brightness(x_test0)
#%%
GAMMA = 2
ALPHA = 0.8 # emphasize FP
BETA = 0.2 # more emphasize on FN

# combo loss
cl_ALPHA = 0.5 # < 0.5 penalises FP more, > 0.5 penalises FN more
cl_BETA = 0.5
CE_RATIO = 0.5 #weighted contribution of modified CE loss compared to Dice loss
e=1e-07
     
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device
class CNNmodel(pl.LightningModule):
    def __init__(self):
        super(CNNmodel,self).__init__()
        self.batch_size=bs
        self.learning_rate = 2e-4
        self.nr_channels = nr_channels
        self.net = smp.Unet(classes=num_class, in_channels=self.nr_channels, activation = 'softmax')
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
#%%s_4_256-epoch=32_4.ckpt
size = 512
bs = 16 #batchsize
classes = [ 1,  2,   4,  5,  6,  7,  8,  9, 10]
num_class = len(classes)
EPOCH = 300
reduce = 8
big_list=[]
training_tiles = 8 #2 or 4
for training_tiles in [2,4,8]:
    param=r"C:\Data\uni\uni\Master\Guided-Research\model_param\plot{}V2\s_{}_{}*".format(size,training_tiles,size)
    
    for i in glob.glob(param):
        print(i)
        #nr_channels=int(i[69:70])#128
        nr_channels = int(i[-6:-5])
        # savepath = os.path.join(r'C:\Data\uni\uni\Master\Guided-Research\results\comb',"{}_{}".format(size,EPOCH))
        # if not os.path.exists(savepath): os.mkdir(savepath)
        print(x_test0.shape,y_test0.shape)
        x_test, y_test = process(x_test0, y_test0, size =size, bslice=False, cl2dim=True, Inf2zero=True)
        x_test, y_test = process(x_test0, y_test0, size =size, bslice=True, cl2dim=True, Inf2zero=True)
        print(x_test.shape, y_test.shape)
        test_loader = myloader(x_test, y_test,nr_channels)
        # model=r"C:\Data\uni\uni\Master\Guided-Research\model_param\s_{}_{}-epoch={}.ckpt".format(training_tiles,size,ep_nr)
        new_model = CNNmodel.load_from_checkpoint(checkpoint_path=i,maplocation="gpu")
        #new_model =new_model.cuda()
        trainer=pl.Trainer(gpus=1)
        result_model = trainer.test(new_model,test_dataloaders=test_loader)
    
        nr_tiles=x_test0.shape[0]
        print("{}_{}_{}".format(training_tiles,size,EPOCH))
    
        numimag = np.sqrt(x_test.shape[0]/nr_tiles).astype(int)
        num = (numimag*numimag+1).astype(int)
        classes = [1,2,4,5,6,7,8,9,10]
        big = np.zeros((numimag*size, numimag*size))
        result = np.zeros((1,size, size))
        
        for plot in range(1,2):
            with torch.no_grad():
                for data in test_loader:
                    images, labels = data 
                    output1 = new_model(images.cuda())
                    p1 = torch.argmax(output1.cpu(), dim = 1).cpu().numpy()
                    result = np.concatenate((result, p1),axis=0)
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
                    y_label[i,j] = dic_label[y_test0[(1-1),i,j]]        
            if plot == 1:
                result_plot = result[1:num,:,:] 
        result_plot= np.moveaxis(result_plot, 0, -1)     
        big = np.zeros((numimag*size, numimag*size))
        big = np.where(y_label!=9,big,9)
        for j in range(numimag):    
                for i in range(numimag):
                    big[j*size: (j+1)*size,i*size: (i+1)*size]= result_plot[:,:,i+j*numimag]
                    
        y_label = np.where(y_label==9,np.nan,y_label)
        big = np.where(np.isnan(y_label),y_label,big)
        big_list.append(big)
#%%save 
import pickle
with open("big_list{}V2.pkl".format(size),"wb") as fp:
    pickle.dump(big_list,fp)
big_list=pickle.load(open('big_listV2.pkl',"rb"))
#%%
cmap = matplotlib.colors.ListedColormap(matplotlib.cm.get_cmap("Set3").colors[:9])
bounds= np.linspace(-.5,8.5,10)
norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
# ticks=[0,1,2,3,4,5,6,7,8]
ticks=[0,1,2,3]

widths1 = [1,1,1,1,1,0.1,1.4]
heights1 = [1,1,1]

fig = plt.figure(figsize=(25,25),constrained_layout=False)

# tick_labels= [0,250,500,750,1000,1250,1500,1750]
tick_labels= [0,500,1000,1500]
titles = ['3 channels','4 channels','5 channels','6 channels', '7 channels']
y_labels=['2 tiles','4 tiles','8 tiles']
gs0 = gridspec.GridSpec(3,7,figure=fig,wspace=0.05,hspace=-0.77,height_ratios=heights1,width_ratios=widths1)
gs1 = gridspec.GridSpec(3,7,figure=fig,height_ratios=[1,2.25,1],width_ratios=[1,1,1,0.9,0.1,0.1,1.2],wspace=0.45)
delta =0.025
x = y = np.arange(-3.0, 3.0, delta)
X, Y = np.meshgrid(x, y)
Z1 = np.exp(-X**2 - Y**2)
Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
Z = (Z1 - Z2) * 2

plotnr=0
for a in range(3):#rows
    for b in range(5):#columns
        # f_plot.annotate('{},{}'.format(a,b), **anno_opts)
        f_plot = fig.add_subplot(gs0[a, b])
        plt.imshow(big_list[plotnr],cmap=cmap,norm=norm)
        f_plot.set_xticklabels([])
        f_plot.set_yticklabels([])
        if a == 2:
            f_plot.get_xaxis().set_ticks(tick_labels)
            f_plot.set_xticklabels(tick_labels,fontsize=14)
        if b == 0:
            f_plot.get_yaxis().set_ticks(tick_labels)
            f_plot.set_yticklabels(tick_labels,fontsize=14)
            f_plot.set_ylabel(y_labels[a],fontsize=16)
        if a ==0:
            f_plot.set_title(titles[plotnr],fontsize=18)
        plotnr+=1
# for i, ax in enumerate(fig.axes):        
        
        
f_val = fig.add_subplot(gs1[1,6])
# f_val=plt.imshow(big_list[2],norm=norm,cmap=cmap)
f_val.set_title('validation',fontsize=18)
f_val.set_yticklabels([])
# pltim = plt.imshow(y_label,norm=norm,cmap=cmap)
pltim = plt.imshow(y_label,norm=norm,cmap=cmap)
f_val.get_xaxis().set_ticks(tick_labels)
f_val.set_xticklabels(tick_labels,fontsize=14)
#colorbar
f_colorbar = fig.add_subplot(gs1[1,5])
plt.colorbar(pltim,cax=f_colorbar,ticks=[0,1,2,3,4,5,6,7,8])
f_colorbar.set_yticklabels(['P1','S1a','S2','P2c','Water','P2b','Other','S3a','S1a'],fontsize=18)
# plt.savefig('C:\\Data\\uni\\uni\\Master\\Guided-Research\\results\\comb\size128V2.png',dpi=160)
plt.show()