#!/user/bin/env python
# -*- coding:utf-8 -*-
import os

import cv2
import nibabel
import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt
import shutil
from torch.utils.data.dataset import Dataset
import torch

def saveNpz(data_dir, target_dir):
    resample_rateZ = 6
    crop_shape = [128,128,32]

    data = nib.load(data_dir)
    img = data.get_fdata()
    img_trans = img.transpose(2,1,0)
    shape = np.array(img_trans.shape)
    shape[0] *= resample_rateZ
    shape = (shape[1],shape[0])
    z_resampled = cv2.resize(img_trans,shape)
    img_resampled = z_resampled.transpose(2,1,0)
    resam_shape= img_resampled.shape

    img_crop = img_resampled[int(resam_shape[0]/2 - crop_shape[0]/2 ):int(resam_shape[0]/2+crop_shape[0]/2),
               int(resam_shape[1] / 2 - crop_shape[1] / 2):int(resam_shape[1] / 2 + crop_shape[1] / 2 ),
               int(resam_shape[2] / 2 - crop_shape[2] / 2): int(resam_shape[2] / 2 + crop_shape[2] / 2 )
               ]
    #normalize
    img_crop = (img_crop - img_crop.min())/(img_crop.max()-img_crop.min())
    np.savez(target_dir,vol = img_crop)





def accumulateImgs(Ddir = os.path.join('.','EDnpz'),Sdir = os.path.join('.','ESnpz'), dir = os.path.join('.')):
    z = []
    if not os.path.exists(Ddir):
        os.makedirs(Ddir)
    if not os.path.exists(Sdir):
        os.makedirs(Sdir)

    for idx in range(1,101):
        configdir = os.path.join(dir,'patient{:03d}'.format(idx),'Info.cfg')
        with open(configdir) as f:
            NumED = int(next(f).split()[-1])
            NumES = int(next(f).split()[-1])

        EDdir = os.path.join(dir,'patient{:03d}'.format(idx),'patient{:03d}_frame{:02d}.nii.gz'.format(idx,NumED))
        ESdir = os.path.join(dir,'patient{:03d}'.format(idx),'patient{:03d}_frame{:02d}.nii.gz'.format(idx,NumES))
        EDtarget = os.path.join(Ddir, 'patient{:03d}_ED.npz'.format(idx))
        EStarget = os.path.join(Sdir, 'patient{:03d}_ES.npz'.format(idx))

        #shutil.copy(EDdir,EDtarget)
        #shutil.copy(ESdir, EStarget)

        saveNpz(EDdir, EDtarget)
        saveNpz(ESdir, EStarget)
        # ED = nib.load(EDdir)
        # ES = nib.load(ESdir)
        # Dz.append(ED.shape[-1])
        # Sz.append(ES.shape[-1])


def showimg(imgs):
    for i in range(imgs.shape[2]):
        plt.subplot(4,8,i+1)
        plt.imshow(imgs[:,:,i],cmap='gray')

    plt.show()

class ED_ES_dataset(Dataset):
    def __init__(self,EDdir,ESDir,idxs):
        super(ED_ES_dataset, self).__init__()
        self.EDdir = EDdir
        self.ESDir = ESDir
        self.idxs = idxs

    def __getitem__(self, item):
        EDdir = os.path.join(self.EDdir,'patient{:03d}_ED.npz'.format(self.idxs[item]))
        ESdir = os.path.join(self.ESdir,'patient{:03d}_ES.npz'.format(self.idxs[item]))
        ED = np.load(EDdir)['vol']
        ES = np.load(ESdir)['vol']
        print(item)
        return ED,ES,ES


    def __len__(self):
        return len(self.idxs)

class ED_ED_dataset(Dataset):
    def __init__(self,EDdir,ESDir,idxs):
        super(ED_ED_dataset, self).__init__()
        self.EDdir = EDdir
        self.idxs = idxs

    def __getitem__(self, item):
        EDdir = os.path.join(self.EDdir,'patient{:03d}_ED.npz'.format(self.idxs[item]))
        ED = np.load(EDdir)['vol']


    def __len__(self):
        return len(self.idxs)

class test(Dataset):
    def __init__(self):
        super(test, self).__init__()


    def __getitem__(self, item):
        print(item)
        return item

    def __len__(self):
        return 100

def getFilename(dir = os.path.join('.','ESnpz')):

    list = os.listdir(dir)
    with open('ESfilename.txt','w') as f:
        for name in list:
            f.write(os.path.join(dir,name)+'\n')

def showImage():
    dir = os.path.join('template.nii.gz')
    data = nib.load(dir)
    nib.viewers.OrthoSlicer3D(data.dataobj).show()


if __name__ == '__main__':
    loader = torch.utils.data.dataloader.DataLoader(test(),batch_size=16,drop_last=True,)
    for item in loader:
        print(item)







