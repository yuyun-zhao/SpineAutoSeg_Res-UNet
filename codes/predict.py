#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 2017
@author: Inom Mirzaev
"""
# 如果是想训练的话运行train，权重文件备份一份，此代码能使权重文件weights.h5.bak发生修改
# __future__模块：print_function代表接下来所有print的用法都按照python3，division代表小数除法。
from __future__ import division, print_function

import cv2
from keras.optimizers import Adam

# gridspec 模块，指定图中子图的位置，指定将放置子图的网格的几何。
# 需要设置网格的行数和列数。可选地，可以调整子图布局参数（例如，左，右等）。
import matplotlib.gridspec as gridspec

# plt 模块，绘制折线图
import matplotlib.pyplot as plt

# 便于SSH远程连接Linux服务器，其实没啥用
plt.switch_backend('agg')

import numpy as np

# os模块：包含普遍的操作系统功能，代码与操作系统无关
import os

# 读取DICOM文件
import SimpleITK as sitk
from skimage.transform import resize

# find_contours 用来检测二值图像的边缘轮廓
from skimage.measure import find_contours
import sys
sys.path.append(os.path.abspath('..'))
# import * 为导入模块中所有函数。
# models 自定义模块，包含三个函数：conv_block、level_block、UNet，用来训练模型。
from codes.models import *

# metrics 自定义模块，计算最终参数，包含六个函数：dice_coef，dice_coef_loss，
# numpy_dice，rel_abs_vol_diff，get_boundary，surface_dist
# dice参数用来评估两个相似的样品，通常在医学影像中用于将输出与医疗应用中的mask进行比较
from codes.metrics import *

from codes.augmenters import *
# train 自定义模块，包含函数：img_resize（对图像进行限制对比度自适应均衡化），
# data_to_array（分开训练集及验证集数据，对图像去噪并保存，合并切片数目并保存），load_data，
# augment_validation_data，keras_fit_generator

def MMS_DSC(mask_pred,name="val"):

    mask = np.load('../data/y_' + name + '.npy')

    scores_DSC = []
    scores_PPV = []
    scores_sen = []

    for i in range(mask.shape[0]):

        y_true = mask[i]
        y_pred = mask_pred[i]

        scores_DSC.append( numpy_dice( y_true, y_pred , axis=None) )
        scores_PPV.append( numpy_ppv( y_true, y_pred ))
        scores_sen.append( numpy_sensitivity( y_true, y_pred ))

    scores_DSC = np.array(scores_DSC)
    scores_PPV = np.array(scores_PPV)
    scores_sen = np.array(scores_sen)

    print('Mean DSC:', scores_DSC.mean() )
    print('Median DSC:', np.median(scores_DSC) )
    print('Std DSC:', scores_DSC.std() )
    print('\n')

    print('Mean PPV:', scores_PPV.mean() )
    print('Median PPV:', np.median(scores_PPV) )
    print('Std PPV:', scores_PPV.std() )
    print('\n')

    print('Mean Sensitivity:', scores_sen.mean() )
    print('Median Sensitivity:', np.median(scores_sen) )
    print('Std Sensitivity:', scores_sen.std() )
    print('\n')


def make_plots(X, y, y_pred, n_best=20, n_worst=20, name='train'):

    #PLotting the results'
    img_rows = X.shape[1]
    img_cols = X.shape[2]

    # 确定一个list，这里range是（1,2,3）
    axis = tuple( range(1, X.ndim ) )

    # 计算dice系数，获得一个scores（每个切片一个分数）
    scores = numpy_dice(y, y_pred, axis=axis )

    # 将分数按照降序划定顺序（从大到小）
    sort_ind = np.argsort( scores )[::-1]

    # 有几张切片是有值的
    indice = np.nonzero( y.sum(axis=axis) )[0]
    #Add some best and worst predictions

    #
    img_list = []

    #
    count = 1

    # 分数从大到小排序
    for ind in sort_ind:

        # 必须是有值的
        if ind in indice:

            # 将图像的序号加进去
            img_list.append(ind)
            count+=1
        if count>n_best:
            break

    # 将这些图像的预测分割mask取出来
    segm_pred = y_pred[img_list].reshape(-1,img_rows, img_cols)

    # 将这些图像的原图像取出来
    img = X[img_list].reshape(-1,img_rows, img_cols)

    # 将这些图像的原本mask取出来
    segm = y[img_list].reshape(-1, img_rows, img_cols).astype('float32')

    ###
    n_cols = 4

    # 统计图像个数（这里是前20个），所以一列就有5个
    n_rows = int(np.ceil(len(img) / n_cols))

    # ？？？
    fig = plt.figure(figsize=[4 * n_cols, int(4 * n_rows)])

    # 安排的子图，4x5
    gs = gridspec.GridSpec(n_rows, n_cols)

    # 依次取图像
    for mm in range(len(img)):

        # 第一个子图
        ax = fig.add_subplot(gs[mm])

        # 直接在子图位置展示图像
        ax.imshow(img[mm])

        # 查找第一张图像的原mask的边界
        contours = find_contours(segm[mm], 0.01, fully_connected='high')

        # 遍历contours，n是0,1,2,3, contour是contours里面的
        for n, contour in enumerate(contours):

            # 画线
            ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color='r')

        # 查找第一张图像的预测的mask的边界
        contours = find_contours(segm_pred[mm], 0.01, fully_connected='high')

        # 遍历contours，n是0,1,2,3, contour是contours里面的
        for n, contour in enumerate(contours):

            # 画线
            ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color='y')


        # 标题名
        ax.axis('image')

        # x轴方向
        ax.set_xticks([])

        # y轴方向
        ax.set_yticks([])

        ax.set_aspect(1)  # aspect ratio of 1

    if name == 'train':
        fig.savefig('../images/best_predictions_train.png', bbox_inches='tight', dpi=300)

    elif name == 'val':
        fig.savefig('../images/best_predictions_val.png', bbox_inches='tight', dpi=300)

    elif name == 'ind_val':
        fig.savefig('../images/best_predictions_ind_val.png', bbox_inches='tight', dpi=300)
    ###


    img_list = []
    count = 1

    # 将序号倒叙
    for ind in sort_ind[::-1]:
        if ind in indice:
            img_list.append(ind)
            count+=1
        # 还是20张
        if count>n_worst:
            break

    segm_pred = y_pred[img_list].reshape(-1,img_rows, img_cols)
    img = X[img_list].reshape(-1,img_rows, img_cols)
    segm = y[img_list].reshape(-1, img_rows, img_cols).astype('float32')


    n_cols= 4
    n_rows = int( np.ceil(len(img)/n_cols) )

    fig = plt.figure(figsize=[ 4*n_cols, int(4*n_rows)] )
    gs = gridspec.GridSpec( n_rows , n_cols )

    for mm in range( len(img) ):

        ax = fig.add_subplot(gs[mm])
        ax.imshow(img[mm] )
        contours = find_contours(segm[mm], 0.01, fully_connected='high')
        for n, contour in enumerate(contours):
            ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color='r')

        contours = find_contours(segm_pred[mm], 0.01, fully_connected='high')
        for n, contour in enumerate(contours):
            ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color='y')

        ax.axis('image')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect(1)  # aspect ratio of 1

    if name == 'train':
        fig.savefig('../images/worst_predictions_train.png', bbox_inches='tight', dpi=300)

    elif name == 'val':
        fig.savefig('../images/worst_predictions_val.png', bbox_inches='tight', dpi=300)

    elif name == 'ind_val':
        fig.savefig('../images/worst_predictions_ind_val.png', bbox_inches='tight', dpi=300)

def predict_result():
    mu = np.load('../data/mu_train.npy')
    sigma = np.load('../data/sigma_train.npy')

    model = get_model(512,256)

    # load images and masks of training set
    fileList_image = os.listdir("../data/final_results/image/")

    images_image = []

    for filename in fileList_image:
        itk_img = sitk.ReadImage("../data/final_results/image/" + filename)
        img = sitk.GetArrayFromImage(itk_img)

        # img = smooth_images(img)

        new_imgs = np.zeros([len(img), 512, 256])

        for mm, imgslice in enumerate(img):
            img_ = imgslice[:, 208:672].reshape(imgslice.shape[0], -1)
            new_imgs[mm] = cv2.resize(img_, (256, 512), interpolation=cv2.INTER_NEAREST)


        # 此处为该病人的所有层MRI 放入预测即可。
        img_eachp = new_imgs.reshape(-1, 512, 256, 1)
        img_eachpatient_ = smooth_images(img_eachp)
        img_eachpatient = (img_eachpatient_ - mu) / sigma

        y_pred_eachpatient = model.predict(img_eachpatient, verbose=1, batch_size=1)
        # y_pred = model.predict(X_val, verbose=1, batch_size=1)

        unresized_pred = y_pred_eachpatient.reshape(-1, 512, 256)

        shape = tuple([unresized_pred.shape[0], img.shape[1], img.shape[2]])
        resized_pred = np.zeros(shape)

        for m in range(len(unresized_pred)):
            resized_nonzero_pred = cv2.resize(unresized_pred[m, :, :], (464,img.shape[1]),
                                              interpolation=cv2.INTER_NEAREST)
            resized_pred[m, :, 208:672] = resized_nonzero_pred

        resized_pred[resized_pred>0.5] = 1.0

        resized_pred = resized_pred.astype('uint16')

        mask = sitk.GetImageFromArray(resized_pred)
        mask.SetOrigin(itk_img.GetOrigin())
        mask.SetDirection(itk_img.GetDirection())
        mask.SetSpacing(itk_img.GetSpacing())

        sitk.WriteImage(mask, '../images/mask_case' + filename[4:-7] + '.nii.gz')

def get_model(img_rows, img_cols):

    # 先建一个跟训练集一模一样的UNet模型
    model = UNet((img_rows, img_cols,1), start_ch=8, depth=7, batchnorm=True, dropout=0.5, maxpool=True, residual=True)

    # 导入之前训练好的权重
    model.load_weights('../data/weights7.h5')                # 这里使用训练出的weight7

    # 对模型进行编译
    model.compile( optimizer = Adam(), loss=dice_coef_loss, metrics=[dice_coef])

    return model

# 检查验证集的准确性
def check_predictions(name = 'train', plot=True ):

    # 判断有没有一个文件夹叫images
    if not os.path.isdir('../images'):
        # 如果没有就新建一个
        os.mkdir('../images')


    # 导入训练集MRI数据
    X_train = np.load('../data/X_train.npy')
    # 导入训练集mask数据
    y_train = np.load('../data/y_train.npy')

    # 导入验证集MRI数据
    X_val = np.load('../data/X_val.npy')
    # 导入验证集mask数据
    y_val = np.load('../data/y_val.npy')

    # 导入测试集MRI数据
    X_ind_val = np.load('../data/X_ind_val.npy')
    # 导入测试集mask数据
    y_ind_val = np.load('../data/y_ind_val.npy')

    # 取图像长宽
    img_rows = X_val.shape[1]
    img_cols = X_val.shape[2]

    # 跳转函数get_model
    model = get_model(img_rows, img_cols)

    # 首先对X_train数据进行预测
    if name == 'train':

        # 如果是train_list说明样本数>10，因此将X_train导入，得到训练结果y_pred
        y_pred = model.predict( X_train, verbose=1, batch_size=1 )

        # 所以训练集的最终结果是
        print('Results on train set:')

        # 跳转numpy_dice函数，将mask与最终训练结果做比较
        print('DSC:', numpy_dice(y_train, y_pred))

        print('PPV:', numpy_ppv(y_train, y_pred))

        print('sensitivity:', numpy_sensitivity(y_train, y_pred))
        print('\n')

        MMS_DSC(y_pred,name=name)

    elif name == 'val':
        # 如果是val_list说明样本数<10，因此将X_val导入，得到训练结果y_pred
        y_pred = model.predict( X_val, verbose=1, batch_size=1)

        # 所以验证集的最终结果是
        print('Results on validation set')

        # 跳转numpy_dice函数，将mask与最终训练结果作比较
        print('DSC:', numpy_dice(y_val, y_pred))

        print('PPV:', numpy_ppv(y_val, y_pred))

        print('sensitivity:', numpy_sensitivity(y_val, y_pred))
        print('\n')

        MMS_DSC(y_pred,name=name)

    else:
        y_pred = model.predict( X_ind_val, verbose=1, batch_size=1)

        # 所以验证集的最终结果是
        print('Results on independent validation set')

        # 跳转numpy_dice函数，将mask与最终训练结果作比较
        print('DSC:', numpy_dice(y_ind_val, y_pred))

        print('PPV:', numpy_ppv(y_ind_val, y_pred))

        print('sensitivity:', numpy_sensitivity(y_ind_val, y_pred))
        print('\n')

        MMS_DSC(y_pred,name=name)


    if plot and name == 'train':
        make_plots(X_train, y_train, y_pred,name=name)  # 将训练集中最好的20张和最坏的20张保存出来

    elif plot and name == 'val':
        make_plots(X_val, y_val, y_pred,name=name) # 将验证集中最好的20张和最坏的20张保存出来

    elif plot and name == 'ind_val':
        make_plots(X_ind_val, y_ind_val, y_pred,name=name) # 将独立验证集中最好的20张和最坏的20张保存出来


if __name__=='__main__':

    # # 检查训练集的准确性
    # check_predictions(name = 'train', plot=False)
    # #
    # # # 检查验证集的准确性
    # check_predictions(name = 'val',plot=False)
    #
    # # 检查独立验证集的准确性
    check_predictions(name = 'ind_val',plot=False)

    # 输出测试集
    predict_result()


