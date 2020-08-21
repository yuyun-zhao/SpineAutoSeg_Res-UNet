
import  cv2
import  numpy as np
from matplotlib import pyplot as plt
import os


import os
import SimpleITK as sitk
import numpy as np
from matplotlib import pyplot as plt
from skimage.exposure import equalize_adapthist, equalize_hist
import cv2
from augmenters import *
import nibabel as nib


# itk_img = sitk.ReadImage('../images/mask_case196.nii')
# img = sitk.GetArrayFromImage(itk_img)
#
# plt.imshow(img[5], cmap='gray')
# plt.show()
#
# a = 1

for i in range(100):
    imag_data = nib.load('../data/final_results/image/Case196.nii.gz').get_data()
    plt.figure(1)
    plt.imshow(imag_data[:,:,i], cmap='gray')

    imag_data = nib.load('../images/Case196.nii').get_data()
    # imag_data = nib.load('../images/mask_case210.nii.gz').get_data()

    plt.figure(2)
    plt.imshow(imag_data[:,:,i], cmap='gray')
    plt.show()
    a = 1
#
# val_image = np.load('../data/X_test.npy')
#
# val_image1 = np.load('F:/2019BMEdata/X_test1.npy')
#
# fileList_test_image = os.listdir("../data/final_results/image/")
# fileList_test_mask = os.listdir("../data/final_results/groundtruth/")


# mu1 = np.load("D:/2019BME_nii/data/mu_train.npy")
# mu2 = np.load("F:/2019BMEdata/sigma_train.npy")
# mu3 = np.load('F:/2019BMEdata/sigma_train1.npy')
#


images_test_image = []

allnonzeromat = []

for filename in  fileList_test_image:
    itk_img = sitk.ReadImage('D:/2019BME_nii/data/final_results/image/' + filename)
    img = sitk.GetArrayFromImage(itk_img)

    plt.imshow(img[1], cmap='gray')
    plt.show()
    #
    a = 1

for filename in  fileList_test_mask:

    itk_img = sitk.ReadImage('../data/final_results/groundtruth/'+filename)
    img = sitk.GetArrayFromImage(itk_img)

    plt.imshow(img[1], cmap='gray')
    plt.show()
    images_test_mask.append(new_imgs)

test_masks = np.concatenate(images_test_mask, axis=0).reshape(-1, 512, 256, 1)
test_masks = test_masks.astype(int)




# val_image = np.load('F:/2019BMEdata/X_test.npy')
# val_mask  =  np.load('F:/2019BMEdata/y_test.npy')

# val_image = np.load('D:/2019BME_nii/final_results/X_test.npy')

# val_image = np.load('D:/2019BME_nii/data/X_val.npy')
val_mask = np.load('F:/2019BMEdata/y_final.npy')


# plt.figure(0)
# plt.imshow(val_image[100].reshape(512,256),'gray')

plt.figure(1)
plt.imshow(val_mask[1].reshape(880,880),'gray')
plt.show()


plt.figure(0)
plt.imshow(val_image[100].reshape(256,128),'gray')

plt.figure(1)
plt.imshow(val_mask[100].reshape(256,128),'gray')
plt.show()


a = 1