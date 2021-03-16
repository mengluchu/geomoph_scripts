import os
import cv2 as cv
from glob import glob
from shutil import move, copyfile
from tqdm import tqdm
from scipy import misc


def clip_image_label(img, label, patch_size, save_path, tif_name, overlap_rate=1/8):
    overlap_len = int(patch_size * overlap_rate)
    stride_len = patch_size - overlap_len
    m, n, _ = img.shape
    tmp_val = (m- overlap_len) // stride_len
    num_m = tmp_val if (m - overlap_len) % stride_len == 0 else tmp_val + 1

    tmp_val = (n - overlap_len) // stride_len
    num_n = tmp_val if (n - overlap_len) % stride_len == 0 else tmp_val + 1
    num = 0
    for i in range(num_m):
        for j in range(num_n):
            if i == num_m - 1 and j != num_n - 1:
                tmp_img = img[-patch_size:, j * stride_len:j * stride_len + patch_size, :]
                tmp_label = label[-patch_size:, j * stride_len:j * stride_len + patch_size]
            elif i != num_m - 1 and j == num_n - 1:
                tmp_img = img[i * stride_len:i * stride_len + patch_size, -patch_size:, :]
                tmp_label = label[i * stride_len:i * stride_len + patch_size, -patch_size:]
            elif i == num_m - 1 and j == num_n - 1:
                tmp_img = img[-patch_size:, -patch_size:, :]
                tmp_label = label[-patch_size:, -patch_size:]
            else:
                tmp_img = img[i * stride_len:i * stride_len + patch_size,
                          j * stride_len:j * stride_len + patch_size, :]
                tmp_label = label[i * stride_len:i * stride_len + patch_size,
                            j * stride_len:j * stride_len + patch_size]
            cv.imwrite(os.path.join(save_path, 'img', tif_name+'_'+str(num)+'.tif'), tmp_img)
            cv.imwrite(os.path.join(save_path, 'label', tif_name+'_'+str(num)+'.tif'), tmp_label)
            num += 1


def clip_data(data_path, save_path, patch_size, overlap_rate=1/8):
    os.makedirs(os.path.join(save_path, 'img'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'label'), exist_ok=True)

    img_path_all = glob(os.path.join(data_path, 'img', '*.tif'))
    for pth in tqdm(img_path_all):
        basename = os.path.basename(pth)
        tif_name = basename.split('.')[0]
        label_path = os.path.join(data_path, 'label', basename)
        img = cv.imread(pth, cv.IMREAD_COLOR)
        label = cv.imread(label_path, cv.IMREAD_COLOR)
        clip_image_label(img, label, patch_size, save_path, tif_name, overlap_rate=overlap_rate)


def label_1to255(label_path):
    label_pathes = glob(os.path.join(label_path, '*.tif'))
    for pth in tqdm(label_pathes):
        label = cv.imread(pth, cv.IMREAD_GRAYSCALE)
        label = label * 255
        cv.imwrite(pth, label)


def rename_dataset(root, fore_name=''):
    img_pathes = glob(os.path.join(root, 'img', '*.tif'))
    for pth in tqdm(img_pathes):
        basename = os.path.basename(pth)
        os.rename(pth, os.path.join(root, 'img', fore_name+'_'+basename))

    label_pathes = glob(os.path.join(root, 'label', '*.tif'))
    for pth in tqdm(label_pathes):
        basename = os.path.basename(pth)
        os.rename(pth, os.path.join(root, 'label', fore_name + '_' + basename))


def rename_(root):
    files = glob(os.path.join(root, '*.*'))
    for pth in tqdm(files):
        basename = os.path.basename(pth)
        newname = basename.replace('label', 'zl')
        os.rename(pth, os.path.join(root, newname))


def check_shape(root):
    imgs = glob(os.path.join(root, 'imgs', '*.tif'))
    for pth in imgs:
        basename = os.path.basename(pth)
        img = cv.imread(pth, cv.IMREAD_COLOR)
        label = cv.imread(os.path.join(root, 'labels', basename), cv.IMREAD_GRAYSCALE)
        ih, iw, _ = img.shape
        lh, lw = label.shape

        if ih != lh or iw!=lw:
            print(basename)
            h = min(ih, lh)
            w = min(iw, lw)
            cv.imwrite(os.path.join(root, 'imgs', basename), img[:h, :w, :])
            cv.imwrite(os.path.join(root, 'labels', basename), label[:h, :w])
        else:
            print(basename, img.shape, label.shape)


if __name__ == '__main__':
    ''''''
