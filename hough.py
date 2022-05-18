# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 15:26:36 2022

软件包：opencv, numpy， math，operator
操作系统：Windows 10 专业版 20H2
CPU：Intel(R) Core(TM) i5-10400F CPU @ 2.90GHz   2.90 GHz
GPU：AMD Radeon RX 5600 XT
内存：2×8GB

@author: 1851604-高若凹
"""

import numpy as np # 矩阵计算库
import math # 数学库
import operator # 用于排序


class Hough_transform:
    def __init__(self, img, angle, threshold=135):
        '''
        img: 输入的边缘图像
        angle: 输入的梯度方向矩阵
        step: Hough 变换步长大小
        threshold: 筛选单元的阈值
        '''
        self.img = img # 输入的边缘图片
        self.angle = angle # 输入的梯度方向矩阵
        self.y, self.x = img.shape[0:2] # 输入图片宽、长
        self.radius = math.ceil(math.sqrt(self.y ** 2 + self.x ** 2)) # 输入图片半径
        self.vote_matrix = np.zeros([self.y, self.x]) # 创建二维投票矩阵
        self.threshold = threshold # 阈值
        self.circles = [] # 用于储存检测出来的圆的半径和圆心坐标

    def Hough_transform_algorithm(self):
        '''
        按照 x,y,radius 建立三维空间，根据图片中边上的点沿梯度方向对空间中的所有单元进行投票。每个点投出来结果为一折线。
        return:  投票矩阵
        '''

        for i in range(1, self.y - 1): # 遍历图像除外围像素外的所有像素
            for j in range(1, self.x - 1):
                if self.img[i][j] > 0: # 如果是边缘则沿边缘画线
                    y = i
                    x = j 
                    # 沿梯度正方向投票
                    while y < self.y and x < self.x and y >= 0 and x >= 0: # 保证像素在图像内
                        self.vote_matrix[y][x] += 1 # 在投票矩阵对应位置加一
                        y = round(y + 1.414*np.sin(self.angle[i][j])) # 圆心y坐标沿梯度方向移动
                        x = round(x + 1.414*np.cos(self.angle[i][j])) # 圆心x坐标向梯度方向移动
                                    
                     # 沿梯度反方向投票
                    y = round(y - 1.414*np.sin(self.angle[i][j])) # 圆心y坐标沿梯度反方向移动
                    x = round(x - 1.414*np.cos(self.angle[i][j])) # 圆心x坐标向左移动

                    while y < self.y and x < self.x and y >= 0 and x >= 0: # 保证圆心在图像内
                        self.vote_matrix[y][x] += 1 # 在投票矩阵对应位置加一
                        y = round(y - 1.414*np.sin(self.angle[i][j])) # 圆心y坐标沿梯度反方向移动
                        x = round(x - 1.414*np.cos(self.angle[i][j])) # 圆心x坐标向梯度反方向移动
                                    
        return self.vote_matrix   # 返回投票矩阵

    def Select_Circle(self):
        '''
        按照阈值从投票矩阵中筛选出合适的圆，并作极大化抑制，这里的非极大化抑制采
        用的是邻近点结果取平均值的方法，而非单纯的取极大值。
        return: None
        '''

		# 挑选投票数大于阈值的圆
        houxuanyuan = [] # 用于储存候选圆心坐标
        # 非最大化抑制
        for i in range(0, self.y - 1): # 遍历投票矩阵所有元素 
            for j in range(0, self.x - 1):
                    if self.vote_matrix[i][j] < self.vote_matrix[i+1][j] or self.vote_matrix[i][j] < self.vote_matrix[i-1][j] \
                        or self.vote_matrix[i][j] < self.vote_matrix[i][j+1] or self.vote_matrix[i][j] < self.vote_matrix[i][j-1] \
                        or self.vote_matrix[i][j] < self.vote_matrix[i+1][j+1] or self.vote_matrix[i][j] < self.vote_matrix[i+1][j-1] \
                        or self.vote_matrix[i][j] < self.vote_matrix[i-1][j+1] or self.vote_matrix[i][j] < self.vote_matrix[i-1][j-1]:
                        self.vote_matrix[i][j] = 0 # 如果不是局部最大则置零
                        
        for i in range(0, self.y - 1): # 遍历投票矩阵所有元素 
            for j in range(0, self.x - 1):
                if self.vote_matrix[i][j] > self.threshold: # 如果该点票数大于阈值，则可能是圆心
                        houxuanyuan.append((j, i)) # 将该圆心信息储存
        if len(houxuanyuan) == 0: # 如果未找到，则输出
            print("No Circle in this threshold.")
            return
        
        x, y = houxuanyuan[0] # 圆坐标
        possible = [] # 储存可能圆坐标
        middle = [] # 储存坐标均值
        for circle in houxuanyuan:
            if abs(x - circle[0]) <= 5 and abs(y - circle[1]) <= 5: # 设定一个误差范围（这里设定方圆5个像素以内，属于误差范围），在这个范围内的圆心视为同一个圆心           
                possible.append([circle[0], circle[1]]) # 储存坐标
            else:
                result = np.array(possible).mean(axis=0) # 对同一范围内的圆心，半径取均值
                middle.append((result[0], result[1])) # 储存坐标均值
                possible.clear() # 清空possible为以便下个循环使用
                possible.append([circle[0], circle[1]]) # 储存坐标
        result = np.array(possible).mean(axis=0)  # 将最后一组同一范围内的圆心，半径取均值
        middle.append((result[0], result[1]))  # 误差范围内的圆取均值后，放入其中
        
        def takeFirst(elem): # 取首值函数
            return elem[0]   
        middle.sort(key=takeFirst)  # 排序 
        
        # 重复类似上述取均值的操作，并将圆逐个输出
        x, y = middle[0] # 取出均值的第一个坐标
        possible = [] # 储存可能圆坐标
        for circle in middle: # 遍历均值圆心 
            if abs(x - circle[0]) <= 10 and abs(y - circle[1]) <= 10: # 设定一个误差范围（这里设定方圆5个像素以内，属于误差范围），在这个范围内的圆心视为同一个圆心
                possible.append([circle[0], circle[1]]) # 储存坐标
            else:
                result = np.array(possible).mean(axis=0) # 对同一范围内的圆心，半径取均值
                self.circles.append((result[0], result[1])) # 保存至圆心列表
                possible.clear() # 清空possible为以便下个循环使用
                possible.append([circle[0], circle[1]])    
        result = np.array(possible).mean(axis=0) # 对同一范围内的圆心，半径取均值
        self.circles.append((result[0], result[1])) # 保存至圆心列表
        
        # 计算半径
        radius = {} # 候选半径字典
        for i in range(0, self.y - 1): # 遍历二值化图片所有像素
            for j in range(0, self.x - 1):
                if self.img[i][j] > 0: # 如果非零则计算与所有候选圆心的距离
                    for h in self.circles: # 遍历所有候选圆
                        distence = round(math.sqrt((abs(h[0]-j) ** 2 + abs(h[1]-i) ** 2))) # 计算两点距离，即可能的半径
                        if distence in radius: # 判断是否已经存在
                            radius[distence] = radius[distence] + 1 # 投票
                        else:
                            radius[distence] = 1 # 若不在，则将其添加到radius中
                            
        radius2 = sorted(radius.items(), key=operator.itemgetter(1), reverse=True) # 从大到小排序
        r = radius2[0][0] # 所得半径
               
        return r


