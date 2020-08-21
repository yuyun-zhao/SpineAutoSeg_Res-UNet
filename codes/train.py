#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 17:18:38 2017
@author: Inom Mirzaev
"""

# 如果是想训练的话运行train，权重文件备份一份，此代码能使权重文件weights.h5.bak发生修改
# __future__模块：print_function代表接下来所有print的用法都按照python3，division代表小数除法。
from __future__ import division, print_function
from collections import defaultdict
from alt_model_checkpoint import AltModelCheckpoint
from keras.utils import multi_gpu_model

# os模块：包含普遍的操作系统功能，代码与操作系统无关
import os, pickle, sys
import shutil

# partial用来传递一个函数
from functools import partial
#from itertools import izip

# opencv
import cv2

# 优化器用的是自适应学习率优化算法，可以对学习率进行实时调节（随着时间的增加逐渐变小、
# 既有二阶矩梯度估计又有一阶矩梯度估计，下降速度快
from keras.optimizers import Adam, SGD

# Callbacks 是一组在训练的特定阶段被调用的函数集，你可以使用Callbacks来观察训练过程中网络内部的状态和统计信息
# ModelCheckpoint 回调类允许你定义检查模型权重的位置，文件应如何命名，以及在什么情况下创建模型的 Checkpoint              ？
from keras.callbacks import ModelCheckpoint

# LearningRateScheduler 可以按照epoch自动调整学习率（这里由于已经使用了Adam优化器可以不用使用这个）
# EarlyStopping 早停法，防止过拟合，例如在验证集的AUC达到一定程度后停止训练
from keras.callbacks import LearningRateScheduler, EarlyStopping

# ImageDataGenerator 图片生成器，同时也可以在批处理中对数据进行增强，扩充数据集大小，
# 比如进行旋转，变形，归一化等，增强模型的泛化能力。
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
from scipy.misc import imresize
from skimage.transform import resize

# equalize_adapthist 限制对比度自适应直方图均衡化
from skimage.exposure import equalize_adapthist, equalize_hist

# import * 为导入模块中所有函数。
# models 自定义模块，包含三个函数：conv_block、level_block、UNet，用来训练模型。                                                                            ？
from models import *

# metrics 自定义模块，计算最终参数，包含六个函数：dice_coef，dice_coef_loss，
# numpy_dice，rel_abs_vol_diff，get_boundary，surface_dist
# dice参数用来评估两个相似的样品，通常在医学影像中用于将输出与医疗应用中的mask进行比较
from metrics import dice_coef, dice_coef_loss

# augmenters 自定义模块，进行数据增强处理，包含两个函数：elastic_transform，smooth_images
from augmenters import *

# Python没有main主函数，Python使用缩进对齐组织代码的执行，
# 所有没有缩进的代码（非函数定义和类定义），都会在载入时自动执行，这些代码，可以认为是Python的main函数。

os.environ["CUDA_VISIBLE_DEVICES"] = "4, 5"

# 这边是加载数据并且把它组织成自己的一个形式
# load_data 将预处理后的数据载入
def load_data():                                                                       # 改了

    X_train = np.load('../data/X_train.npy')
    y_train = np.load('../data/y_train.npy')
    X_val = np.load('../data/X_val.npy')
    y_val = np.load('../data/y_val.npy')

    return X_train, y_train, X_val, y_val

#   2.真实输入：图片大小为256*256；n_imgs 图片数量为15x10^4；batch_size 批大小为32；
def keras_fit_generator(img_rows=96, img_cols=96, n_imgs=10**4, batch_size=32, regenerate=True):

    if regenerate:

        # 2.运行data_to_array,输入img_rows、img_cols为256,数据预处理后放入data文件夹目录下备用
        data_to_array(img_rows, img_cols)
        #preprocess_data()

    # 6.跳转load_data，输入训练集数据、验证集数据
    X_train, y_train, X_val, y_val = load_data()
    # 三维结构分别为（切片数目累加和，影像长，影像宽）


    # 将影像长宽取出赋给img_rows、img_cols
    # X_val, y_val = augment_validation_data(X_val, y_val, seed=10)
    img_rows = X_train.shape[1]
    img_cols =  X_train.shape[2]

    # Provide the same seed and keyword arguments to the fit and flow methods

    # 这是一些参数，其实这些参数在这里可以不用改，这里的参数都是keras提供的传统的数据增强的方式
    # 转多少角度、平移多少位置，是不是对称的，都是经典的数据增强方式

    # range()返回的是range object，而np.nrange()返回的是numpy.adarray()
    # 两者都是均匀地（evenly）等分区间；
    # range尽可用于迭代，而np.arange可用作迭代，也可用做向量。
    # range()不支持步长为小数，np.arange()支持步长为小数，
    # 如果只有一个参数默认起点为0。

    # np.meshgrid 根据传入的2个一维数组参数生成2个数组元素的列表，将前后组合起来形成网格点矩阵
    # indexing影响meshgrid()函数返回的矩阵的表示形式，变成ndarray结构
    # 返回x为将x横向量向下复制img_cols行，返回y为将y列向量向右复制img_rows行，形成变成ndarray结构
    x, y = np.meshgrid(np.arange(img_rows), np.arange(img_cols), indexing='ij')

    # functools.partial 偏函数，这里是仿射变换函数，用来对图像做仿射和弹性变换
    elastic = partial(elastic_transform, x=x, y=y, alpha=img_rows*1.5, sigma=img_rows*0.07 )

    # we create two instances with the same arguments
    # dict 创建字典，参数均为经典的数据增强方式
    data_gen_args = dict(
        featurewise_center=False,                   # 布尔值，使输入数据集去中心化（均值为0）, 按feature执行
        featurewise_std_normalization=False,        # 布尔值，将输入除以数据集的标准差以完成标准化, 按feature执行。
        rotation_range=10.,                         # 整数，数据提升时图片随机转动的角度。随机选择图片的角度（0-180）
        width_shift_range=0.1,                      # 浮点数，图片宽度的某个比例，数据提升时图片随机水平偏移的幅度。
        height_shift_range=0.1,                     # 浮点数，图片高度的某个比例，数据提升时图片随机竖直偏移的幅度。
        horizontal_flip=True,                       # 布尔值，随机水平翻转。随机图片水平翻转（水平翻转不影响图片语义）。
        vertical_flip=True,                         # 布尔值，进行随机竖直翻转。
        zoom_range=[1, 1.2],                        # 浮点数或形如[lower,upper]的列表，随机缩放的幅度，若为浮点数，
                                                    #   则相当于[lower,upper] = [1 - zoom_range, 1+zoom_range]。
                                                    #   用来进行随机的放大。
        fill_mode='constant',                       # ‘constant’，‘nearest’，‘reflect’或‘wrap’之一，
                                                    #   当进行变换时超出边界的点将根据本参数给定的方法进行处理
        preprocessing_function=elastic)             # 将上文仿射变换函数应用起来。该函数将在任何其他修改之前运行。
                                                    #   该函数接受一个参数，为一张图片（秩为3的numpy array），
                                                    #   并且输出一个具有相同shape的numpy array

    # ImageDataGenerator Keras中函数，用来数据扩充，增加训练数据，防止过拟合，
    # 将这些经典参数填入函数，并在接下来将图像填入函数，可得到相应的扩充后的训练集图像与训练集mask
    # **data_gen_args的目的是存放变量参数，等待之后填充（固定用法）
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    seed = 2
    # 设置种子并将训练集图像和训练集mask使用fit函数填入
    image_datagen.fit(X_train, seed=seed)
    mask_datagen.fit(y_train, seed=seed)

    # flow 生成一个迭代器，接收numpy数组和标签为参数，生成经过数据提升或标准化后的batch数据，
    # 并在一个无限循环中不断的返回batch数据
    image_generator = image_datagen.flow(X_train, batch_size=batch_size, seed=seed)
    mask_generator = mask_datagen.flow(y_train, batch_size=batch_size, seed=seed)

    # zip 两两组合，压缩后留下备用
    train_generator = zip(image_generator, mask_generator)
    
    # 这边是unet网络的描述，先看参数前面是两个尺度，灰度图通道是1，开始输入8通道
    # 深度是7最里面这层有1024个chanel，再看一下，dropout是0.5，残差模块连起来

    # 7.建立训练模型，需要输入开始图像的长宽及其他参数，自动生成UNet模型，以下都还在陆陆续续填参数进模型
    model = UNet((img_rows, img_cols,1), start_ch=8, depth=7, batchnorm=True, dropout=0.5, maxpool=True, residual=True)
   # model = multi_gpu_model(model, gpus=2, by_name=)
    
    
   # model.load_weights('../data/weights3.h5')

    # 输出训练模型各层的参数状况进行核对
    model.summary()



    # filepath = "model_{epoch:02d}-{val_acc:.2f}.hdf5"
    # checkpoint = ModelCheckpoint(os.path.join(save_dir, filepath), monitor='val_acc', verbose=1,
    #                              save_best_only=True)



    # ModelCheckpoint 保存最佳模型
    # （1）monitor：监视值，当监测值为val_acc时，mode自动为max，当监测值为val_loss时，mode自动为min。
    # （2）save_best_only：当设置为True时，只有监测值有改进时才会保存当前的模型
    model_checkpoint = AltModelCheckpoint(
        '../data/weights7.h5',model, monitor='val_loss', save_best_only=True)


    # ？？？
    c_backs = [model_checkpoint]

    # EarlyStopping 早停法，防止过拟合，例如在验证集的AUC达到一定程度后停止训练，或loss函数开始增加时停止训练
    # （1）monitor：有’acc’,’val_acc’,’loss’,’val_loss’等等。正常情况下如果有验证集，就用
    # ’val_acc’或者’val_loss’。如果用的是交叉检验就用’acc’，这里明明有验证集为什么要用'loss'？
    # （2）min_delta：增大或减小的阈值，只有大于这个部分才算作improvement。
    # 这个值的大小取决于monitor，也反映了你的容忍程度。小于这个值的都漠不关心。
    # （3）patience：能够容忍多少个epoch内都没有improvement。这个设置其实是在抖动和真正的准确率下降之间做tradeoff。
    # 如果patience设的大，那么最终得到的准确率要略低于模型可以达到的最高准确率。
    # 如果patience设的小，那么模型很可能在前期抖动，还在全图搜索的阶段就停止了，准确率一般很差。
    # patience的大小和learning rate直接相关。在learning rate设定的情况下，前期先训练几次观察抖动的epoch number，
    # 比其稍大些设置patience。在learning rate变化的情况下，建议要【略小于】最大的抖动epoch number。
    c_backs.append( EarlyStopping(monitor='loss', min_delta=0.001, patience=5) )

    model = multi_gpu_model(model, gpus=2)
    
    # Adam 自适应学习率优化算法，可以对学习率进行实时调节（随着时间的增加逐渐变小、
    # 既有二阶矩梯度估计又有一阶矩梯度估计，下降速度快，loss设定为dice函数（医学影像处理二分类问题较好）
    model.compile(optimizer=Adam(lr=0.001), loss=dice_coef_loss, metrics=[dice_coef])


    model.fit_generator(
                        train_generator,                                # generator：生成器函数

                        steps_per_epoch=n_imgs//batch_size,             # steps_per_epoch：整数，当生成器返回steps_per_epoch
                                                                        # 次数据时计一个epoch结束，执行下一个epoch

                        epochs=20,                                      # epochs：整数，数据迭代的轮数

                        verbose=2,                                      # verbose：日志显示，0为不在标准输出流输出日志信息，
                                                                        # 1为输出进度条记录，2为每个epoch输出一行记录

                        shuffle=True,                                   # shuffle：布尔值，是否随机打乱数据，默认为True
                        validation_data=(X_val, y_val),
                        # validation_data=(np.concatenate([X_train,X_val]), np.concatenate([y_train,y_val]) ),
                                                                        # validation_data：验证集生成器？连起来？？？

                        callbacks=c_backs,
                        use_multiprocessing=True)                      # ？？？

# use_multiprocessing=True

# 当.py文件被直接运行时，if __name__ == '__main__'之下的代码块将被运行；
# 当.py文件以模块形式被导入时，if __name__ == '__main__'之下的代码块不被运行。
# 主函数从此开始：
if __name__=='__main__':
    # time 时间模块，可用来计时
    import time

    start = time.time()

    # 1.运行keras_fit_generator 数据分批传送入内存
    #   batch_size 批大小为32；n_imgs 图片数量为15x10^4（有待考证）；
    #   图片大小为256*256；regenerate表示图片分成验证集和测试集、去噪，把一系列的数据保存起来
    keras_fit_generator(img_rows=512, img_cols=256, regenerate=False,
                       n_imgs=15*10**4, batch_size=32)
    # keras_fit_generator(img_rows=256, img_cols=256, regenerate=False,
    #                     n_imgs=15*10**1, batch_size=5)
    end = time.time()
    #输出时间
    print('Elapsed time:', round((end-start)/60, 2 ) )

# （1）batchsize：批大小。在深度学习中，一般采用SGD训练，即每次训练在训练集中取batchsize个样本训练
# （2）iteration：1个iteration等于使用batchsize个样本训练一次；
# （3）epoch：1个epoch等于使用训练集中的【全部】样本训练一次，通俗的讲epoch的值就是整个数据集被轮几次




