from __future__ import division, print_function

from keras import backend as K
K.set_image_data_format('channels_last')
import numpy as np
from scipy.ndimage import morphology

def dice_coef(y_true, y_pred, smooth=1.0):

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (
        K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def ppv_coef(y_true, y_pred, smooth=1.0):

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (K.sum(y_pred_f) + smooth)

def sensitivity_coef(y_true, y_pred, smooth=1.0):

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (K.sum(y_true_f) + smooth)

def numpy_dice(y_true, y_pred, axis=None, smooth=1.0):

    intersection = y_true*y_pred

    return ( 2. * intersection.sum(axis=axis) +smooth)/ (np.sum(y_true, axis=axis) + np.sum(y_pred, axis=axis) +smooth )

def numpy_ppv(y_true, y_pred, axis=None, smooth=1.0):

    intersection = y_true*y_pred

    return ( intersection.sum(axis=axis) +smooth)/ (np.sum(y_pred, axis=axis) +smooth )

def numpy_sensitivity(y_true, y_pred, axis=None, smooth=1.0):

    intersection = y_true*y_pred

    return (intersection.sum(axis=axis) +smooth)/ (np.sum(y_true, axis=axis)+smooth )


