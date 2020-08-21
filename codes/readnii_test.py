
import skimage

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
fileList_test_image = os.listdir("../data/test/image/")
fileList_test_mask = os.listdir("../data/test/groundtruth/")

images_test_image = []

allnonzeromat = []

for filename in  fileList_test_image:
    itk_img = sitk.ReadImage('D:/2019BME_nii/data/test/image/' + filename)
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
        # #
        # a = 1

    images_test_image.append(new_imgs)
    allnonzeromat.append(nonzeromat)

test_image = np.concatenate(images_test_image, axis=0).reshape(-1, 256, 128, 1)
test_images_raw = smooth_images(test_image)

test_images = (test_images_raw - mu) / sigma

np.save('F:/2019BMEdata/X_test.npy', test_images)

test_nonzeromat = np.array(allnonzeromat)
np.save('F:/2019BMEdata/test_nonzeromat.npy', test_nonzeromat)



images_test_mask = []

# test_nonzeromat = np.load('../data/test_nonzeromat.npy')

i = 0

for filename in  fileList_test_mask:

    itk_img = sitk.ReadImage('../data/test/groundtruth/'+filename)
    img = sitk.GetArrayFromImage(itk_img)

    new_imgs = np.zeros([len(img), 256, 128])
    nonzeromat = test_nonzeromat[i]

    i += 1

    for mm, imgslice in enumerate(img):

        nonzero = tuple(nonzeromat[mm])
        img = imgslice[:, nonzero].reshape(imgslice.shape[0], -1)

        new_imgs[mm] = cv2.resize(img, (128, 256), interpolation=cv2.INTER_NEAREST)

        # aaa = new_imgs[mm]
        #
        # plt.imshow(aaa, 'gray')
        # plt.show()

    images_test_mask.append(new_imgs)

test_masks = np.concatenate(images_test_mask, axis=0).reshape(-1, 256, 128, 1)
test_masks = test_masks.astype(int)

np.save('F:/2019BMEdata/y_test.npy', test_masks)


