# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 22:00:34 2022

软件包：opencv, numpy, os
操作系统：Windows 10 专业版 20H2
CPU：Intel(R) Core(TM) i5-10400F CPU @ 2.90GHz   2.90 GHz
GPU：AMD Radeon RX 5600 XT
内存：2×8GB

@author: 1851604-高若凹
"""

import numpy as np # 导入矩阵运算库
import os # 导入文件处理库
import cv2  # 利用opencv进行图像处理

picname = ['gaussian', 'origin', 'pepper'] # 三张图片名称集合

input_dir = './images/' # 输入图片文件夹
out_dir = './task1/' # 输出图片文件夹
doc = os.listdir(input_dir) # 所有图片文件名称列表

pic = [] # 定义空的数组用以储存图像矩阵

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))  # 此处声明结构元素

def img_erode(img, kernel, center_coo, numb, mode): # 定义腐蚀变换函数
    '''
    img: 待处理图像
    kernel: 结构化元素
    center_coo: 结构化元素原点位置
    numb: 保存图片名称代号
    mode: 模式类型
    '''

    kernel_w = kernel.shape[0] # 结构化元素的宽
    kernel_h = kernel.shape[1] # 结构化元素的长
    
    if kernel[center_coo[0], center_coo[1]] == 0: # 确定原点坐标是否在结构化元素内
        raise ValueError("指定原点不在结构元素内！")  # 若不在则报错
    
    erode_img = np.zeros(shape=img.shape) # 定义与原图相同大小的空矩阵用于存放结果
    
    for i in range(center_coo[0], img.shape[0]-kernel_w+center_coo[0]+1): 
        for j in range(center_coo[1], img.shape[1]-kernel_h+center_coo[1]+1): # 遍历原图每个像素，令结构化元素原点与该像素重合
            a = img[i-center_coo[0]:i-center_coo[0]+kernel_w, j-center_coo[1]:j-center_coo[1]+kernel_h] # 找到原图与结构化元素重合的小矩阵
            
            erode_img[i, j] = a.min() #将该矩阵的最小像素值存放在结果图对应位置
    
    if mode == 1: # 模式切换
        cv2.imwrite(out_dir + '3x3_Erode_' + picname[numb] + '.png', erode_img) # 保存图像
    else:
        return erode_img # 返回结果
    
def img_dilate(img, kernel, center_coo, numb, mode): # 定义膨胀变换函数
    '''
    img: 待处理图像
    kernel: 结构化元素
    center_coo: 结构化元素原点位置
    numb: 保存图片名称代号
    mode: 模式类型
    '''
    
    kernel_w = kernel.shape[0] # 结构化元素的宽
    kernel_h = kernel.shape[1] # 结构化元素的长
    
    if kernel[center_coo[0], center_coo[1]] == 0: # 确定原点坐标是否在结构化元素内
        raise ValueError("指定原点不在结构元素内！") # 若不在则报错
   
    dilate_img = np.zeros(shape=img.shape) # 定义与原图相同大小的空矩阵用于存放结果
    
    for i in range(center_coo[0], img.shape[0]-kernel_w+center_coo[0]+1):
        for j in range(center_coo[1], img.shape[1]-kernel_h+center_coo[1]+1): # 遍历原图每个像素，令结构化元素原点与该像素重合
            a = img[i-center_coo[0]:i-center_coo[0]+kernel_w, j-center_coo[1]:j-center_coo[1]+kernel_h] # 找到原图与结构化元素重合的小矩阵
            
            dilate_img[i, j] = a.max() #将该矩阵的最大像素值存放在结果图对应位置
    
    if mode == 1: # 模式切换
        cv2.imwrite(out_dir + '3x3_Dilate_' + picname[numb] + '.png', dilate_img) # 保存图像
    else:
        return dilate_img # 返回结果

def img_tophat(img, kernel, center_coo, numb, mode): # 定义顶帽变换函数
    '''
    img: 待处理图像
    kernel: 结构化元素
    center_coo: 结构化元素原点位置
    numb: 保存图片名称代号
    mode: 模式类型
    '''
    # 开运算
    img1 = img_erode(img, kernel, center_coo, numb, mode) # 腐蚀
    img2 = img_dilate(img1, kernel, center_coo, numb, mode) # 膨胀
    result = img - img2 # 顶帽变换
    
    cv2.imwrite(out_dir + '3x3_Tophat_' + picname[numb] + '.png', result) # 保存图像

def img_blackhat(img, kernel, center_coo, numb, mode): # 定义黑帽变换函数
    '''
    img: 待处理图像
    kernel: 结构化元素
    center_coo: 结构化元素原点位置
    numb: 保存图片名称代号
    mode: 模式类型
    '''
    # 闭运算
    img1 = img_dilate(img, kernel, center_coo, numb, mode) # 膨胀
    img2 = img_erode(img1, kernel, center_coo, numb, mode) # 腐蚀
    result = img2 - img # 黑帽运算
    
    cv2.imwrite(out_dir + '3x3_Blackhat_' + picname[numb] + '.png', result) # 保存图像

for file in doc: 
    I = cv2.imread(input_dir+file, 0) # 依次读取全部灰度图片
    pic.append(I) # 储存灰度图像矩阵
    #cv2.imwrite(out_dir + file, I) # 保存灰度图像
    
for i in range(len(pic)): # 对所有图像进行滤波
    img_erode(pic[i], kernel, (1,1), i, 1) # 腐蚀
    img_dilate(pic[i], kernel, (1,1), i, 1) # 膨胀
    img_tophat(pic[i], kernel, (1,1), i, 0) # 顶帽
    img_blackhat(pic[i], kernel, (1,1), i, 0) # 黑帽
    
    
    
    
    