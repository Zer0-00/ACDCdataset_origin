#!/user/bin/env python
# -*- coding:utf-8 -*-
import os

import cv2
import nibabel
import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt
import skimage.transform
from torch.utils.data.dataset import Dataset
import scipy.ndimage.interpolation as inter
import time

def saveNpz(data_dir, target_dir):
    resampled_pix = [1.5,1.5,2.5]
    crop_shape = [128,128,32]

    data = nib.load(data_dir)
    img = data.get_fdata()
    norm = (img - img.min()) / (img.max() - img.min())
    resampled_factor = [data.header['pixdim'][i+1]/resampled_pix[i] for i in range(3)]
    resampled = inter.zoom(norm,resampled_factor)
    resam_shape = resampled.shape

    if resam_shape[2] < 32:
        print(data_dir,data.header['pixdim'],resam_shape)

    crop = [int(np.floor((resam_shape[i]-crop_shape[i])/2)) for i in range(3)]


    cropped = resampled[crop[0]:crop[0]+crop_shape[0],crop[1]:crop[1]+crop_shape[1],crop[2]:crop[2]+crop_shape[2]]
    #print(data_dir,data.header['pixdim'],data.shape,resam_shape,cropped.shape)
    #print(data_dir,data.header.get_best_affine())
    showimg(cropped)

    #np.savez(target_dir,vol=cropped)







def accumulateImgs(Ddir = os.path.join('.','EDnpz_test'),Sdir = os.path.join('.','ESnpz_test'), dir = os.path.join('.')):
    z = []
    if not os.path.exists(Ddir):
        os.makedirs(Ddir)
    if not os.path.exists(Sdir):
        os.makedirs(Sdir)

    #plt.ion()
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
        #fig1 = plt.figure(1)
        #fig1.suptitle('patient{}'.format(idx))
        saveNpz(EDdir, EDtarget)
        #plt.figure(2)
        saveNpz(ESdir, EStarget)
        #plt.show()
        #plt.pause(0.01)
        #ED = nib.load(EDdir)
        # ES = nib.load(ESdir)
        # Dz.append(ED.shape[-1])
        # Sz.append(ES.shape[-1])
        #nib.viewers.OrthoSlicer3D(ED.dataobj).show()


def showimg(imgs):
    for i in range(imgs.shape[2]):
        plt.subplot(4,8,i+1)
        plt.imshow(imgs[:,:,i],cmap='gray')

def getFilename(dir = os.path.join('.','ESnpz_test')):

    list = os.listdir(dir)
    with open('ESfilename_test.txt','w') as f:
        for name in list:
            f.write(os.path.join(dir,name)+'\n')

def showImage():
    dir = 'patient101/patient101_frame01.nii.gz'
    data = nib.load(dir)
    print(data.header['pixdim'])
    nib.viewers.OrthoSlicer3D(data.dataobj).show()


if __name__ == '__main__':
    accumulateImgs()







