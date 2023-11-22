#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 16:57:08 2023

@author: shineber
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

ori_path = './红外弱小目标提取样例/'
imgs = os.listdir(ori_path) 
print('共{}张图片'.format(len(imgs)))

#读取图片显示一下
image2 = cv2.imread(ori_path+imgs[1], 0)
plt.imshow(image2,cmap='gray')
plt.show()

#梯度运算
kernel = np.ones((7, 7), np.uint8)
gradient = cv2.morphologyEx(image2,cv2.MORPH_GRADIENT,kernel)
plt.imshow(gradient,cmap='gray')
plt.show()

#二值化处理
_, binary_image2 = cv2.threshold(gradient, 90, 255, cv2.THRESH_BINARY)
plt.subplot(121),plt.imshow(binary_image2,cmap='gray')
# binary_image_rgb = cv2.cvtColor(binary_image2, cv2.COLOR_GRAY2RGB)
# 寻找轮廓
contours, hierarchy = cv2.findContours(binary_image2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(contours)
# 绘制矩形框
#for contour in contours:
#    x, y, w, h = cv2.boundingRect(contour)
#    cv2.rectangle(binary_image2, (x, y), (x+w, y+h), (0, 255, 255), 4)
cv2.rectangle(binary_image2, (60, 60), (190, 190), (0, 255, 255), 3)
plt.subplot(122),plt.imshow(binary_image2)

cv2.imshow('Original Image', binary_image2)

cv2.waitKey(0)
cv2.destroyAllWindows()