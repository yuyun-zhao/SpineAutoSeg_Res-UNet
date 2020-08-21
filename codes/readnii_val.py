

import os
import SimpleITK as sitk
import numpy as np
from matplotlib import pyplot as plt
from skimage.exposure import equalize_adapthist, equalize_hist
import cv2
from augmenters import *

mu = np.load('F:/2019BMEdata/mu_train.npy')
sigma = np.load('F:/2019BMEdata/sigma_train.npy')


# load images and masks of training set
fileList_val_image = os.listdir("../data/val/image/")
fileList_val_mask = os.listdir("../data/val/groundtruth/")

images_val_image = []

allnonzeromat = []

for filename in  fileList_val_image:
    itk_img = sitk.ReadImage('D:/2019BME_nii/data/val/image/' + filename)
    img = sitk.GetArrayFromImage(itk_img)

    # img = smooth_images(img)

    new_imgs = np.zeros([len(img), 256, 128])
    nonzeromat = []

    for mm, imgslice in enumerate(img):

        nonzero = np.where(imgslice[0]>0)
        img = imgslice[:, nonzero].reshape(imgslice.shape[0],-1)

        nonzeromat.append(nonzero)

        # img01 = (img-np.min(img))/(np.max(img)-np.min(img))

        # img = equalize_adapthist(img01, clip_limit=0.05)

        # plt.subplot(121)
        # plt.imshow(img01, cmap='gray')
        #
        # plt.subplot(122)
        # plt.imshow(img3, cmap='gray')
        #
        # plt.show()

        new_imgs[mm] = cv2.resize(img, (128, 256), interpolation=cv2.INTER_NEAREST)
        #
        # plt.imshow(new_imgs[mm], cmap='gray')
        # plt.show()
        #
        # a = 1

    images_val_image.append(new_imgs)
    allnonzeromat.append(nonzeromat)

val_image = np.concatenate(images_val_image, axis=0).reshape(-1, 256, 128, 1)
val_images_raw = smooth_images(val_image)

val_images = (val_images_raw - mu) / sigma

np.save('F:/2019BMEdata/X_val.npy', val_images)

val_nonzeromat = np.array(allnonzeromat)
np.save('F:/2019BMEdata/val_nonzeromat.npy', val_nonzeromat)




images_val_mask = []
#
# val_nonzeromat = np.load('../data/val_nonzeromat.npy')

i = 0

for filename in  fileList_val_mask:

    itk_img = sitk.ReadImage('../data/val/groundtruth/'+filename)
    img = sitk.GetArrayFromImage(itk_img)

    new_imgs = np.zeros([len(img), 256, 128])
    nonzeromat = val_nonzeromat[i]

    i += 1

    for mm, imgslice in enumerate(img):

        nonzero = nonzeromat[mm]
        img = imgslice[:, nonzero].reshape(imgslice.shape[0], -1)

        new_imgs[mm] = cv2.resize(img, (128, 256), interpolation=cv2.INTER_NEAREST)

        # plt.imshow(new_imgs[mm].reshape(512, 256), 'gray')
        # plt.show()

    images_val_mask.append(new_imgs)

val_masks = np.concatenate(images_val_mask, axis=0).reshape(-1, 256, 128, 1)
val_masks = val_masks.astype(int)

np.save('F:/2019BMEdata/y_val.npy', val_masks)


