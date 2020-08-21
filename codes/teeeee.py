
import  cv2
import  numpy as np
from matplotlib import pyplot as plt

# mask = np.arange(18).reshape(3,6)
#
# print(mask[:, 0:2])
#
# st = 'case196.nii.gz'
#
# print(st[4:-7])
#
# print(mask[0])
#
# print(mask.shape[0])
#
#
# # val_image = np.load('F:/2019BMEdata/X_test.npy')
# # val_mask  =  np.load('F:/2019BMEdata/y_test.npy')
#
# # val_image = np.load('D:/2019BME_nii/final_results/X_test.npy')
#
# val_image = np.load('D:/2019BME_nii/data_clahe/X_val.npy')
# val_mask = np.load('D:/2019BME_nii/data_clahe/y_val.npy')
#
#
# plt.figure(0)
# plt.imshow(val_image[100].reshape(256,256),'gray')
#
# plt.figure(1)
# plt.imshow(val_mask[100].reshape(256,256),'gray')
# plt.show()
#
#
# plt.figure(0)
# plt.imshow(val_image[100].reshape(256,128),'gray')
#
# plt.figure(1)
# plt.imshow(val_mask[100].reshape(256,128),'gray')
# plt.show()
#
#
# a = 1


plt.plot(list(range(0,5*len(cost_plot_train),5)), cost_plot_train)
plt.plot(list(range(0,5*len(cost_plot_val),5)), cost_plot_val)

plt.xlabel('epoch')
plt.ylabel('cost')
plt.legend(['training set cost', 'validation set cost'])

plt.show()



