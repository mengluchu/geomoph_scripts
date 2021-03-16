from torch import nn
import torch
import warnings
import os
from torch.autograd import Variable
import numpy as np
import cv2 as cv
from data_input_for_net import LoadTest
from glob import glob
from baseline.fcn import FCN8s
from models.mfmsNet import *
from models.mfcnet2 import *
from models.mfcnet3 import *


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.cuda.current_device()
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
warnings.filterwarnings('ignore')


def predictWithOverlapB(model, img, patch_size=512, overlap_rate=1/4, isroad=False):
    '''

    :param model: a trained model
    :param img: a path for an image
    :param patch_size:
    :param overlap_rate:
    :return:
    '''
    # subsidiary value for the prediction of an image with overlap
    boder_value = int(patch_size * overlap_rate / 2)
    double_bv = boder_value * 2
    stride_value = patch_size - double_bv
    most_value = stride_value + boder_value

    # an image for prediction
    # img = cv.imread(img_path, cv.IMREAD_COLOR)
    m, n, _ = img.shape
    load_data = LoadTest()
    if max(m, n) <= patch_size:
        tmp_img = img
        tmp_img = load_data(tmp_img)
        with torch.no_grad():
            tmp_img = Variable(tmp_img)
            tmp_img = tmp_img.to(DEVICE).unsqueeze(0)
            result = model(tmp_img)
        output = result if not isinstance(result, (list, tuple)) else result[0]
        output = output if isroad else F.sigmoid(output)
        pred = output.data.cpu().numpy().squeeze(0).squeeze(0)  # [0]
        # pred = pred * 255
        pred[pred >= 0.5] = 255
        pred[pred < 0.5] = 0
        return pred.astype(np.uint8)
    else:
        tmp = (m - double_bv) // stride_value  # 剔除重叠部分相当于无缝裁剪
        new_m = tmp if (m - double_bv) % stride_value == 0 else tmp + 1
        tmp = (n - double_bv) // stride_value
        new_n = tmp if (n - double_bv) % stride_value == 0 else tmp + 1
        FullPredict = np.zeros((m, n), dtype=np.uint8)
        for i in range(new_m):
            for j in range(new_n):
                if i == new_m - 1 and j != new_n - 1:
                    tmp_img = img[
                              -patch_size:,
                              j * stride_value:((j + 1) * stride_value + double_bv), :]
                elif i != new_m - 1 and j == new_n - 1:
                    tmp_img = img[
                              i * stride_value:((i + 1) * stride_value + double_bv),
                              -patch_size:, :]
                elif i == new_m - 1 and j == new_n - 1:
                    tmp_img = img[
                              -patch_size:,
                              -patch_size:, :]
                else:
                    tmp_img = img[
                              i * stride_value:((i + 1) * stride_value + double_bv),
                              j * stride_value:((j + 1) * stride_value + double_bv), :]
                tmp_img = load_data(tmp_img)
                with torch.no_grad():
                    tmp_img = Variable(tmp_img)
                    tmp_img = tmp_img.to(DEVICE).unsqueeze(0)
                    result = model(tmp_img)
                output = result if not isinstance(result, (list, tuple)) else result[0]
                output = output if isroad else F.sigmoid(output)
                pred = output.data.cpu().numpy().squeeze(0).squeeze(0)  # [0]

                pred[pred >= 0.5] = 255
                pred[pred < 0.5] = 0

                if i == 0 and j == 0:  # 左上角
                    FullPredict[0:most_value, 0:most_value] = pred[0:most_value, 0:most_value]
                elif i == 0 and j == new_n-1:  # 右上角
                    FullPredict[0:most_value, -most_value:] = pred[0:most_value, boder_value:]
                elif i == 0 and j != 0 and j != new_n - 1:  # 第一行
                    FullPredict[0:most_value, boder_value + j * stride_value:boder_value + (j + 1) * stride_value] = \
                        pred[0:most_value, boder_value:most_value]

                elif i == new_m - 1 and j == 0:  # 左下角
                    FullPredict[-most_value:, 0:most_value] = pred[boder_value:, :-boder_value]
                elif i == new_m - 1 and j == new_n - 1:  # 右下角
                    FullPredict[-most_value:, -most_value:] = pred[boder_value:, boder_value:]
                elif i == new_m - 1 and j != 0 and j != new_n - 1:  # 最后一行
                    FullPredict[-most_value:, boder_value + j * stride_value:boder_value + (j + 1) * stride_value] = \
                        pred[boder_value:, boder_value:-boder_value]

                elif j == 0 and i != 0 and i != new_m - 1:  # 第一列
                    FullPredict[boder_value + i * stride_value:boder_value + (i + 1) * stride_value, 0:most_value] = \
                        pred[boder_value:-boder_value, 0:-boder_value]
                elif j == new_n - 1 and i != 0 and i != new_m - 1:  # 最后一列
                    FullPredict[boder_value + i * stride_value:boder_value + (i + 1) * stride_value, -most_value:] = \
                        pred[boder_value:-boder_value, boder_value:]
                else:  # 中间情况
                    FullPredict[
                    boder_value + i * stride_value:boder_value + (i + 1) * stride_value,
                    boder_value + j * stride_value:boder_value + (j + 1) * stride_value] = \
                        pred[boder_value:-boder_value, boder_value:-boder_value]
        return FullPredict


class Test(object):
    def __init__(self, save_path, imgpath, weight_path):
        self.imgpath = imgpath
        self.weight_path = weight_path
        self.save_path = save_path + '/predict/'
        os.makedirs(self.save_path, exist_ok=True)

    def predict(self):
        '''
        :return:
        '''
        img_pathes = glob(self.imgpath + '/*.tif')
        # model = get_refinenet(input_size=512, num_classes=1, pretrained=False)
        # model = DeepLabv3_plus(in_channels=3, num_classes=1, backend='resnet101', os=16)
        model = MFCNetO2()
        model.load_state_dict(torch.load(self.weight_path))
        model.to(DEVICE)
        model.eval()
        for i, path in enumerate(img_pathes):
            basename = os.path.basename(path)
            print('正在预测:%s, 已完成:(%d/%d)' % (basename, i, len(img_pathes)))
            img = cv.imread(path, cv.IMREAD_COLOR)
            pred = predictWithOverlapB(model, img, patch_size=1024)
            cv.imwrite(self.save_path + basename, pred)
        print('预测完毕!')


if __name__ == '__main__':

    root = '/home/zl/dataset/water_body_dataset/aerial_dataset/MFCNetO2'
    save_path = '/home/zl/dataset/water_body_dataset/aerial_dataset/MFCNetO2-predict'
    img_path = '/home/zl/dataset/water_body_dataset/aerial_dataset/test-data/img'
    weight_path = root + '/Epoch_27_TrainLoss_0.1100_Valacc_0.9918.pkl'
    predict_fuc = Test(save_path, img_path, weight_path)
    predict_fuc.predict()
