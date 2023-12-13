import os
import os.path
import numpy as np
import random
import h5py
import torch
import cv2
import glob
import torch.utils.data as udata
from utils import data_augmentation
from pathlib import Path

def normalize(data):
    return data/255.

def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw-win+0+1:stride, 0:endh-win+0+1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win*win,TotalPatNum], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[:,i:endw-win+i+1:stride,j:endh-win+j+1:stride]
            Y[:,k,:] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])

def prepare_data(data_path, patch_size, stride, aug_times=1):
    # train
    if True:
        print('process training data')
        # scales = [1, 0.9, 0.8, 0.7]
        scales = [1]
        # files_i = glob.glob(os.path.join(data_path, 'res_input', '*.png'))
        # files_o = glob.glob(os.path.join(data_path, 'res_output', '*.png'))
        # files_i.sort()
        # files_o.sort()
        h5f = h5py.File('train.h5', 'w')
        train_num = 0
        list_j = [11, 4, 3]
        for j in list_j:
            files_i = glob.glob(os.path.join(data_path, f'{j}_', 'res_input', '*.png'))
            files_o = glob.glob(os.path.join(data_path, f'{j}_', 'res_output', '*.png'))
            files_i.sort()
            files_o.sort()

            for i in range(len(files_i)):
                name = Path(files_i[i]).stem
                img_i = cv2.imread(files_i[i])
                img_o = cv2.imread(files_o[i])
                h, w, c = img_i.shape
                print(h,w,c)

                if np.all(img_i==0) and np.all(img_o==0):
                    continue

                Img_i = np.expand_dims(img_i[:,:,0].copy(), 0)
                Img_i = np.float32(normalize(Img_i))
                Img_o = np.expand_dims(img_o[:,:,0].copy(), 0)
                Img_o = np.float32(normalize(Img_o))
                # data = [Img_i.copy(), Img_o.copy()]
                data = [Img_i, Img_o]
                h5f.create_dataset(str(j)+str(name), data=data)
                train_num += 1
        
        h5f.close()
        print('training set, # samples %d\n' % train_num)
        # val
        print('\nprocess validation data')
        files_i.clear()
        files_o.clear()
    files_i = glob.glob(os.path.join(data_path, '2_', 'res_input', '*.png'))
    files_o = glob.glob(os.path.join(data_path, '2_', 'res_output', '*.png'))
    files_w = glob.glob(os.path.join(data_path, '2_', 'res_wnr', '*.png'))
    files_i.sort()
    files_o.sort() 
    files_w.sort() 
    h5f = h5py.File('val.h5', 'w')
    val_num = 0
    for i in range(len(files_i)):
        name = Path(files_i[i]).stem
        img_i = cv2.imread(files_i[i])
        img_o = cv2.imread(files_o[i])
        img_w = cv2.imread(files_w[i])
        img_i = np.expand_dims(img_i[:,:,0], 0)
        img_o = np.expand_dims(img_o[:,:,0], 0)
        img_w = np.expand_dims(img_w[:,:,0], 0)
        img_i = np.float32(normalize(img_i))
        img_o = np.float32(normalize(img_o))
        img_w = np.float32(normalize(img_w))
        img = [img_i, img_o, img_w]
        # h5f.create_dataset(str(val_num), data=img)
        h5f.create_dataset(str(name), data=img)
        val_num += 1
    h5f.close()
    print('val set, # samples %d\n' % val_num)

class Dataset(udata.Dataset):
    def __init__(self, train=True):
        super(Dataset, self).__init__()
        self.train = train
        if self.train:
            h5f = h5py.File('train.h5', 'r')
        else:
            h5f = h5py.File('val.h5', 'r')
        self.keys = list(h5f.keys())
        random.shuffle(self.keys)
        h5f.close()
    def __len__(self):
        return len(self.keys)
    def __getitem__(self, index):
        if self.train:
            h5f = h5py.File('train.h5', 'r')
        else:
            h5f = h5py.File('val.h5', 'r')
        key = self.keys[index]
        data = np.array(h5f[key])
        name = h5f[key].name
        h5f.close()
        return [torch.Tensor(data), name]
