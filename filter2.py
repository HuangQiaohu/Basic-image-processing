# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 10:33:26 2022

软件包：opencv, numpy, os， math
操作系统：Windows 10 专业版 20H2
CPU：Intel(R) Core(TM) i5-10400F CPU @ 2.90GHz   2.90 GHz
GPU：AMD Radeon RX 5600 XT
内存：2×8GB

@author: 1851604-高若凹
"""

import os # 导入文件处理库
import cv2 # 导入图像处理库
import math # 导入数学库
import numpy as np # 导入矩阵处理库

pic = [] # 定义空的数组用以储存图像矩阵

input_dir = './images/' # 输入图片文件夹
out_dir = './task2/' # 输出图片文件夹
doc = os.listdir(input_dir) # 所有图片文件名称列表

picname = ['gaussian', 'origin', 'pepper'] # 三张图片名称集合

def CutImg(img, h, w, ksize): # 定义图片裁剪函数，将卷积前填充的多余像素裁剪
    '''
    img: 待处理图像
    h,w : 原图像长，宽
    ksize: 卷积核大小
    '''
        
    n = ksize//2 # 根据卷积核大小确定裁剪范围
    result = img[n:h+n, n:w+n] # 裁剪
    return result # 返回结果

def GaussianKernel(ksize, sigmac): # 定义高斯核生成函数
    '''
    ksize: 高斯核尺寸
    sigmac: 用来控制图像灰度变化权重
    '''
    
    r = ksize // 2 # 高斯核半径
    c = r 
    denominator = 2 * sigmac * sigmac # 高斯公式中e的幂的分母
    kernel = np.zeros((ksize, ksize)) # 生成一个空矩阵用于储存高斯核的值
    
    for i in range(-r, r + 1): # 遍历高斯核中的每个元素
        for j in range(-c, c + 1): 
            kernel[i + r][j + c] = math.exp(-(i * i + j * j) / denominator) # 计算每个元素的值
   
    return kernel # 得到并返回高斯核

def BilateralFilter(img, ksize, sigmac, sigmas,numb): # 定义双边滤波函数
    '''
    img: 待处理图像
    ksize: 权重矩阵尺寸
    sigmac: 用来控制图像灰度变化权重
    sigmas: 用来控制空间距离变化权重
    numb: 保存图片名称代号
    '''
    
    gkernel = GaussianKernel(ksize, sigmac) # 得到高斯核
    radius = ksize // 2 # 权值矩阵半径
    h, w = img.shape # 获得单通道灰度图像长宽
    result = np.zeros((h, w)) # 构建空矩阵储存结果图片   
    
    for ri in range(radius, h - radius): # 遍历原图片每个像素
        for ci in range(radius, w - radius):
            # 根据下标获得该处理模板区域内的像素值
            start_x, end_x = ci - radius, ci + radius # x方向区域范围
            start_y, end_y = ri - radius, ri + radius # y方向区域范围
            region = img[start_y:end_y + 1, start_x:end_x + 1] # 获得该区域像素值
            similarity_weight = np.exp(-0.5 * np.power(region - img[ri, ci], 2.0) / math.pow(sigmas, 2)) # 计算像素相似度
            
            weight = similarity_weight * gkernel # 最终权值矩阵
            weight = weight / np.sum(weight) # 归一化
            
            result[ri, ci] = np.sum(region * weight) # 得到滤波后的像素
            
    result = CutImg(result, 374, 1238, ksize) # 裁剪图像
    
    cv2.imwrite(out_dir + '7x7_Bilateral_' + picname[numb] + '.png', result) # 储存图像 


def GuideFilter(gimg, img, ksize, eps, numb): # 定义导向滤波函数
    '''
    gimg: 导向图片
    img: 待处理图像
    ksize: 滤波器尺寸
    eps: 正则化参数
    numb: 保存图片名称代号
    '''
    
    gimg1 = gimg/255.0 # 导向图片归一化
    img1 = img/255.0 # 输入图片归一化  
    
    mean_g = cv2.blur(gimg1, ksize) # 导向图片的均值平滑        
    mean_i = cv2.blur(img1, ksize) # 输入图片的均值平滑   
    mean_gg = cv2.blur(gimg1*gimg1, ksize) # 导向图片×导向图片的均值平滑    
    mean_gi = cv2.blur(gimg1*img1, ksize) # 导向图片×输入图片的均值平滑
    
    var_g = mean_gg - mean_g * mean_g # 计算窗口中的方差   
    cov_gi = mean_gi - mean_g * mean_i # 计算窗口中的协方差
   
    a = cov_gi / (var_g + eps) # 计算线性相关因子a
    b = mean_i - a*mean_g # 计算线性相关因子b
        
    mean_a = cv2.blur(a, ksize) # 对a进行均值平滑
    mean_b = cv2.blur(b, ksize) # 对b进行均值平滑
    
    q = mean_a*gimg1 + mean_b # 计算导向滤波结果
    q = CutImg(q, 374, 1238, ksize[0]) # 裁剪图像
    
    # 因为导向滤波返回的是灰度值范围在[0,1]之间的图像矩阵，保存8位图要先乘255，再转换数据类型
    result = q * 255 # 变换成可视图像
    result [result > 255] = 255 # 像素将最大值设定为255
    result = np.round(result) # 每个像素取整
    result = result.astype(np.uint8) # 转换成uint8格式  
    
    cv2.imwrite(out_dir + '7x7_Guide_' + picname[numb] + '.png', result) # 保存图像
    

def FastGuideFilter(gimg, img, ksize, eps, s, numb): # 定义快速导向滤波函数
    '''
    gimg: 导向图片
    img: 待处理图像
    ksize: 滤波器尺寸
    eps: 正则化参数
    s: 缩小系数
    numb: 保存图片名称代号
    '''
    
    gimg1 = gimg/255.0 # 导向图片归一化
    img1 = img/255.0 # 输入图片归一化
    
    h, w = gimg1.shape[:2] # 输入图像的高、宽

    # 缩小图像以减少采样像素点
    size = (int(round(w * s)), int(round(h * s))) # 图片缩小后的尺寸
    small_g = cv2.resize(gimg1, size, interpolation=cv2.INTER_CUBIC) # 将导向图片缩小
    small_i = cv2.resize(img1, size, interpolation=cv2.INTER_CUBIC) # 将输入图片缩小

    # 缩小滤波器尺寸
    X = ksize[0] # 原滤波器尺寸
    small_ksize = (int(round(X * s)), int(round(X * s))) # 将原滤波器缩小得到新滤波器
   
    mean_small_g = cv2.blur(small_g, small_ksize) # 缩小导向图片的均值平滑
    mean_small_i = cv2.blur(small_i, small_ksize) # 缩小输入图片的均值平滑

    mean_small_gg = cv2.blur(small_g * small_g, small_ksize) # 缩小导向图片×缩小导向图片的均值平滑
    mean_small_gi = cv2.blur(small_g * small_i, small_ksize) # 缩小导向图片×缩小输入图片的均值平滑

    var_small_g = mean_small_gg - mean_small_g * mean_small_g # 计算缩小窗口中的方差
    cov_small_gi = mean_small_gi - mean_small_g * mean_small_i # 计算缩小窗口中的协方差

    small_a = cov_small_gi / (var_small_g + eps) # 计算线性相关因子a
    small_b = mean_small_i - small_a * mean_small_g # 计算线性相关因子b
 
    mean_small_a = cv2.blur(small_a, small_ksize) # 对a进行均值平滑
    mean_small_b = cv2.blur(small_b, small_ksize) # 对b进行均值平滑

    # 放大至原图片大小
    size1 = (w, h) # 原图片尺寸
    mean_a = cv2.resize(mean_small_a, size1, interpolation=cv2.INTER_LINEAR) # 将缩小的线性相关因子a恢复至原尺寸
    mean_b = cv2.resize(mean_small_b, size1, interpolation=cv2.INTER_LINEAR) # 将缩小的线性相关因子b恢复至原尺寸

    q = mean_a * gimg1 + mean_b # 计算导向滤波结果
    q = CutImg(q, 374, 1238, ksize[0]) # 裁剪图像
    
    # 因为导向滤波返回的是灰度值范围在[0,1]之间的图像矩阵，保存8位图要先乘255，再转换数据类型
    result = q * 255 # 变换成可视图像
    result [result > 255] = 255 # 像素将最大值设定为255
    result = np.round(result) # 每个像素取整
    result = result.astype(np.uint8) # 转换成uint8格式
      
    cv2.imwrite(out_dir + '7x7_FastGuide_' + picname[numb] + '.png', result) # 保存图像

def GaussianFilter(img, numb): # 定义高斯滤波函数
    '''
    img: 待处理图像
    numb: 保存图片名称代号
    '''
    
    gau = cv2.GaussianBlur(img, (7, 7), 0, 0) # 使用7×7高斯滤波器滤波
    
    gau = CutImg(gau, 374, 1238, 7) # 裁剪图像
    
    cv2.imwrite(out_dir + '7x7_Gauss_' + picname[numb] + '.png', gau) # 保存图像

def MeanFilter(img, numb): # 定义均值滤波函数
    '''
    img: 待处理图像
    numb: 保存图片名称代号
    '''
    
    m = cv2.blur(img,(7,7)) # 7×7卷积核均值滤波
    
    m = CutImg(m, 374, 1238, 7) # 裁剪图像
    
    cv2.imwrite(out_dir + '7x7_Mean_' + picname[numb] + '.png', m) # 保存图像
    
    '''
    7×7 均值滤波算子
    [1  1  1  1  1  1  1
     1  1  1  1  1  1  1
     1  1  1  1  1  1  1
     1  1  1  1  1  1  1
     1  1  1  1  1  1  1
     1  1  1  1  1  1  1
     1  1  1  1  1  1  1]
    '''

for file in doc: 
    I = cv2.imread(input_dir+file, 0) # 依次读取全部灰度图片
    L = cv2.copyMakeBorder(I, 3, 3, 3, 3, cv2.BORDER_REPLICATE) # 根据边缘像素值向外扩充3个相同的像素
    pic.append(L) # 储存灰度图像矩阵
    cv2.imwrite(out_dir + file, I) # 保存灰度图像
    
for i in range(len(pic)): # 对所有图像进行滤波
    BilateralFilter(pic[i], 7, 25*25, 25/2, i) # 进行权重矩阵为7×7的双边滤波
    GuideFilter(pic[i], pic[i], (7, 7), 0.01, i) # 进行导向图片与输入图片相同的导向滤波
    #FastGuideFilter(pic[i], pic[i], (7, 7), 0.01, 0.5, i) # 进行导向图片与输入图片相同的快速导向滤波
    MeanFilter(pic[i],i) # 进行均值滤波
    GaussianFilter(pic[i], i) # 进行高斯滤波



     
    
