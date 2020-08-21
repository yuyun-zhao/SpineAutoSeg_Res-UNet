
import skimage

import os
import SimpleITK as sitk
import numpy as np
from matplotlib import pyplot as plt
from skimage.exposure import equalize_adapthist, equalize_hist
import cv2
from augmenters import *


# load images and masks of training set
fileList_train_image = os.listdir("../data/train/image/")
fileList_train_mask = os.listdir("../data/train/groundtruth/")

images_train_image = []
allnonzeromat = []

for filename in  fileList_train_image:
    itk_img = sitk.ReadImage('D:/2019BME_nii/data/train/image/' + filename)
    img = sitk.GetArrayFromImage(itk_img)

    # img = smooth_images(img)

    new_imgs = np.zeros([len(img), 256, 128])
    nonzeromat = []

    for mm, imgslice in enumerate(img):

        nonzero = np.where(imgslice[0]>0)
        img = imgslice[:, nonzero].reshape(imgslice.shape[0],-1)

        nonzeromat.append(nonzero)

        new_imgs[mm] = cv2.resize(img, (128, 256), interpolation=cv2.INTER_NEAREST)


    images_train_image.append(new_imgs)
    allnonzeromat.append(nonzeromat)

train_image = np.concatenate(images_train_image, axis=0).reshape(-1, 256, 128, 1)
train_images_raw = smooth_images(train_image)

mu = np.mean(train_images_raw)
sigma = np.std(train_images_raw)

np.save('F:/2019BMEdata/mu_train.npy', mu)
np.save('F:/2019BMEdata/sigma_train.npy', sigma)


train_images = (train_images_raw - mu) / sigma

train_nonzeromat = np.array(allnonzeromat)

np.save('F:/2019BMEdata/X_train.npy', train_images)
np.save('F:/2019BMEdata/train_nonzeromat.npy', train_nonzeromat)






images_train_mask = []

# train_nonzeromat = np.load('../data/train_nonzeromat.npy')

i = 0

for filename in  fileList_train_mask:

    itk_img = sitk.ReadImage('../data/train/groundtruth/'+filename)
    img = sitk.GetArrayFromImage(itk_img)

    new_imgs = np.zeros([len(img), 256, 128])
    nonzeromat = train_nonzeromat[i]

    i += 1

    for mm, imgslice in enumerate(img):

        nonzero = nonzeromat[mm]
        img = imgslice[:, nonzero].reshape(imgslice.shape[0], -1)

        new_imgs[mm] = cv2.resize(img, (128, 256), interpolation=cv2.INTER_NEAREST)

        # plt.imshow(new_imgs[mm].reshape(512, 256), 'gray')
        # plt.show()

    images_train_mask.append(new_imgs)

train_masks = np.concatenate(images_train_mask, axis=0).reshape(-1, 256, 128, 1)
train_masks = train_masks.astype(int)

np.save('F:/2019BMEdata/y_train.npy', train_masks)
