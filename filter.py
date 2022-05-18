# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 22:10:06 2022

软件包：opencv, numpy, os
操作系统：Windows 10 专业版 20H2
CPU：Intel(R) Core(TM) i5-10400F CPU @ 2.90GHz   2.90 GHz
GPU：AMD Radeon RX 5600 XT
内存：2×8GB

@author: 1851604-高若凹
"""

import cv2  # 利用opencv进行图像处理
import numpy as np # 利用np进行图像矩阵运算

import os # 导入文件处理库

picname = ['gaussian', 'origin', 'pepper'] # 三张图片名称集合

input_dir = './images/' # 输入图片文件夹
out_dir = './task1/' # 输出图片文件夹
doc = os.listdir(input_dir) # 所有图片文件名称列表

pic = [] # 定义空的数组用以储存图像矩阵

def CutImg(img, h, w, ksize): # 定义图片裁剪函数，将卷积前填充的多余像素裁剪
    '''
    img: 待处理图像
    h,w : 原图像长，宽
    ksize: 卷积核大小
    '''
        
    n = ksize//2 # 根据卷积核大小确定裁剪范围
    result = img[n:h+n, n:w+n] # 裁剪
    return result # 返回结果

def Sobel(img, numb): # 定义sobel滤波函数
    '''
    img: 待处理图像
    numb: 保存图片名称代号
    '''
    
    x = cv2.Sobel(img,cv2.CV_16S,1,0, ksize=3) # 用3×3的sobel算子进行x方向滤波
    y = cv2.Sobel(img,cv2.CV_16S,0,1, ksize=3) # y方向滤波
 
    absX = cv2.convertScaleAbs(x)  # 转回uint8编码
    absY = cv2.convertScaleAbs(y)
    
    absX = CutImg(absX, 374, 1238, 3) # 裁剪图像
    absY = CutImg(absY, 374, 1238, 3)
    
    cv2.imwrite(out_dir + '3x3_Sobel_x_' + picname[numb] + '.png', absX) # 保存图像
    cv2.imwrite(out_dir + '3x3_Sobel_y_' + picname[numb] + '.png', absY)
       
    '''
    3×3 Sobel算子
    X方向:[-1 0 +1          Y方向:[-1 -2 -1
           -2 0 +2                 0  0  0   
           -1 0 +1]               +1 +2 +1]
    '''

def DerivativeFilter(img,numb): # 定义导数滤波函数
    '''
    img: 待处理图像
    numb: 保存图片名称代号
    '''
    
    xkernel = np.array(([0, 0, 0],[-0.5, 0, 0.5],[0, 0, 0])) # x方向3×3导数滤波算子
    ykernel = np.array(([0, -0.5, 0],[0, 0, 0],[0, 0.5, 0])) # y方向
    
    dstx = cv2.filter2D(img, -1, xkernel, (1,1)) # x方向导数滤波
    dsty = cv2.filter2D(img, -1, ykernel, (1,1)) # y方向 
    
    dstx = CutImg(dstx, 374, 1238, 3) # 裁剪图像
    dsty = CutImg(dsty, 374, 1238, 3)
    
    cv2.imwrite(out_dir + '3x3_Derivative_x_' + picname[numb] + '.png', dstx) # 保存图像
    cv2.imwrite(out_dir + '3x3_Derivative_y_' + picname[numb] + '.png', dsty)
    
    '''
    3×3 导数滤波算子
    X方向:[0  0  0          Y方向:[0 -1  0 
          -1 +1  0                0 +1  0   
           0  0  0]               0  0  0]
    '''
    
def MedianFilter(img, numb): #定义中值滤波函数
    '''
    img: 待处理图像
    numb: 保存图片名称代号
    '''
    
    med = cv2.medianBlur(img, 3) # 卷积核尺寸为3的中值滤波
    
    med = CutImg(med, 374, 1238, 3) # 裁剪图像
    
    cv2.imwrite(out_dir + '3x3_Median_' + picname[numb] + '.png', med) # 保存图像
    
def MeanFilter(img, numb): # 定义均值滤波函数
    '''
    img: 待处理图像
    numb: 保存图片名称代号
    '''
    
    m = cv2.blur(img,(3,3)) # 3×3卷积核均值滤波
    
    m = CutImg(m, 374, 1238, 3) # 裁剪图像
    
    cv2.imwrite(out_dir + '3x3_Mean_' + picname[numb] + '.png', m) # 保存图像
    
    '''
    3×3 均值滤波算子
    [1  1  1
     1  1  1
     1  1  1]
    '''
    
def GaussianFilter(img, numb): # 定义高斯滤波函数
    '''
    img: 待处理图像
    numb: 保存图片名称代号
    '''
    
    gau = cv2.GaussianBlur(img, (3, 3), 0, 0) # 使用3×3高斯滤波器滤波
    
    gau = CutImg(gau, 374, 1238, 3) # 裁剪图像
    
    cv2.imwrite(out_dir + '3x3_Gauss_' + picname[numb] + '.png', gau) # 保存图像

for file in doc: 
    I = cv2.imread(input_dir+file, 0) # 依次读取全部灰度图片
    L = cv2.copyMakeBorder(I, 1, 1, 1, 1, cv2.BORDER_REPLICATE) # 根据边缘像素值向外扩充1个相同的像素
    pic.append(L) # 储存灰度图像矩阵
    cv2.imwrite(out_dir + file, I) # 保存灰度图像
    
for i in range(len(pic)): # 对所有图像进行滤波
    Sobel(pic[i], i) # Sobel滤波
    DerivativeFilter(pic[i], i) # 导数滤波
    MedianFilter(pic[i], i) # 中值滤波
    MeanFilter(pic[i], i) # 均值滤波
    GaussianFilter(pic[i], i) # 高斯滤波
    










    
    
 

    
    

