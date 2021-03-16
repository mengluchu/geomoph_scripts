# code使用说明：

#训练文件
./train.py使用时，需要手动指定Train(data_path, save_path, patch_size=512, epoch=100, batchSize=2)类相关参数。  

data_path:指定训练数据和验证数的路径，该路径下训练样本和验证样本的目录格式如下：  
训练样本路径：
./data_path/train_data/img  
./data_path/train_data/label
验证集样本路径
./data_path/val_data/img  
./data_path/val_data/label  
注意影像的格式是tif格式，如是其他格式可手动更改代码对应位置

训练时的超参, 例如学习率、batchsize等参数设置：  
epoch：64（本文设置32左右）   
learning rate:1e-4（固定学习率）  
batchsize: 4或者更大 (显存要求在24G以上，进行多卡训练时，采用成倍的增加)

# 测试或预测文件
./predict.py

# 精度评价文件
./metric_cal.py

# 其它
代码有啥问题，请多多交流哇~
