# -*- coding: utf-8 -*-


import numpy as np # 矩阵计算库
import cv2 # 利用opencv进行图像处理

def get_gradient_and_direction(image):
    """ 计算灰度图像梯度与梯度方向
    使用sobel算子求导
         -1 0 1        -1 -2 -1
    Gx = -2 0 2   Gy =  0  0  0
         -1 0 1         1  2  1
    输入
        image: 灰度图
    输出
        gradients: 梯度图像
        direction: 梯度方向
    """
    Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) # x方向sobel算子
    Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) # y方向sobel算子

    W, H = image.shape # 图像宽、长
    gradients = np.zeros([W - 2, H - 2]) # 创建空矩阵储存梯度信息
    direction = np.zeros([W - 2, H - 2]) # 创建空矩阵储存梯度方向信息

    for i in range(W - 2): # 遍历原图像每一个像素
        for j in range(H - 2):
            dx = np.sum(image[i:i+3, j:j+3] * Gx) # 原图像部分与x方向sobel算子卷积得到x方向导数
            dy = np.sum(image[i:i+3, j:j+3] * Gy) # 原图像部分与y方向sobel算子卷积得到y方向导数
            gradients[i, j] = np.sqrt(dx ** 2 + dy ** 2) # 将梯度信息储存
            if dx == 0:
                direction[i, j] = np.pi / 2 # 如果dx = 0 在梯度方向为90°
            else:
                direction[i, j] = np.arctan(dy / dx) # 其他情况下，计算梯度方向，并储存到梯度方向矩阵

    gradients = np.uint8(gradients) # 将梯度图像转化为unit8格式
    
    return gradients, direction

def NMS(gradients, direction):
    """ 非最大值抑制
    输入:
        gradients: 梯度图像
        direction: 梯度方向图像

    输出:
        非最大值抑制后的图像
    """
    W, H = gradients.shape # 梯度图图像宽，长
    nms = np.copy(gradients[1:-1, 1:-1]) # 创建非极大值抑制矩阵

    for i in range(1, W - 1): # 遍历梯度图像每个像素
        for j in range(1, H - 1):
            theta = direction[i, j] # 对应位置梯度方向
            weight = np.tan(theta) # 对应梯度方向的tan值
            if theta > np.pi / 4: # 如果梯度方向大于45小于90度，N、S和NE、SW插值，且tan值为weight倒数
                d1 = [0, 1] # 
                d2 = [1, 1] # 
                weight = 1 / weight
            elif theta >= 0: # 如果梯度方向大于0小于45度，N、S和NE、SW插值，且tan值为weight
                d1 = [1, 0]
                d2 = [1, 1]
            elif theta >= - np.pi / 4: # 如果梯度方向小于0大于45度，N、S和NW、SE插值，且tan值为weight的相反数
                d1 = [1, 0]
                d2 = [1, -1]
                weight *= -1
            else: # 如果梯度方向小于-45大于-90度，N、S和NW、SE插值，且tan值为weight负倒数
                d1 = [0, -1]
                d2 = [1, -1]
                weight = -1 / weight
                
            # 不同情况下的插值系数
            g1 = gradients[i + d1[0], j + d1[1]]
            g2 = gradients[i + d2[0], j + d2[1]]
            g3 = gradients[i - d1[0], j - d1[1]]
            g4 = gradients[i - d2[0], j - d2[1]]

            grade_count1 = g1 * weight + g2 * (1 - weight) # Gp1插值结果
            grade_count2 = g3 * weight + g4 * (1 - weight) # Gp2插值结果

            if grade_count1 > gradients[i, j] or grade_count2 > gradients[i, j]: # 如果该像素点 的梯度小于Gp1和Gp2，则该点一定不是边缘，置零。
                nms[i - 1, j - 1] = 0
                
    return nms

def double_threshold(nms, threshold1, threshold2):
    """ 双阈值处理和滞后边界跟踪
   
    输入:
        nms: 非最大值抑制图像
        threshold1: 弱边界阈值
        threshold2: 强边界阈值

    输出:
        双阈值处理图像
    """
    visited = np.zeros_like(nms) # 遍历过的点集合
    output_image = nms.copy() # 创建输出图片矩阵
    W, H = output_image.shape # 获取输入图片宽，长

    def dfs(i, j):    
        '''深度优先算法，搜索连通边界
        
        输入：
            像素坐标
        输出：
            连通点集
        '''
        
        if i >= W or i < 0 or j >= H or j < 0 or visited[i, j] == 1: # 如果像素不在图片范围内，或该点已经被储存在visited中，则返回
            return
        visited[i, j] = 1 # 若该点在图片中且未被遍历，则将对应位置储存在visited中
        if output_image[i, j] > threshold1: # 如果该店梯度大于弱边界阈值，则进行滞后边界处理
            #output_image[i, j] = 255 # 二值化，置一
            dfs(i-1, j-1) # 对相邻八个像素递归
            dfs(i-1, j)
            dfs(i-1, j+1)
            dfs(i, j-1)
            dfs(i, j+1)
            dfs(i+1, j-1)
            dfs(i+1, j)
            dfs(i+1, j+1)
        else:
            output_image[i, j] = 0 # 如果小于弱边界阈值则一定不是边界，置零

    for w in range(W): # 遍历每一个像素
        for h in range(H):
            if visited[w, h] == 1: # 如果该像素被遍历过则跳过
                continue
            if output_image[w, h] >= threshold2: # 如果该像素点梯度大于强边界阈值则一定是边界
                dfs(w, h) # 进行深度优先算法，搜索连通域
            elif output_image[w, h] <= threshold1: # 如果该像素点梯度大于强边界阈值则一定不是边界
                output_image[w, h] = 0 # 置零
                visited[w, h] = 1 # 同时将该点存入visited中

    for w in range(W): # 遍历每一个像素
        for h in range(H):
            if visited[w, h] == 0: # 如果该点没有被遍历过，则一定是弱边界
                output_image[w, h] = 0 # 置零
                 
    return output_image
           
if __name__ == "__main__":

    input_dir = './img/' # 输入图片文件夹
    out_dir = './task1/' # 输出图片文件夹
    
    image = cv2.imread(input_dir+'lanes.png', 0) # 读取灰度图片
    # 第一步高斯滤波
    smoothed_image = cv2.GaussianBlur(image, (5,5), 0) # 高斯滤波核尺寸为3*3，sigma = 2
    
    # 第二步求图像梯度
    gradients, direction = get_gradient_and_direction(smoothed_image)
    
    cv2.imwrite(out_dir + 'gradient_amplitude.png', gradients) # 保存图片
    
    # 第三步非极大值抑制
    nms = NMS(gradients, direction)
    
    cv2.imwrite(out_dir + 'non_maximum_suppression.png', nms) # 保存图像
    
    #第四步双阈值处理和滞后边界跟踪
    output_image = double_threshold(nms, 100, 220)
    
    cv2.imwrite(out_dir + 'double_threshold_and_connection.png', output_image) # 保存图像
    
    #第五步二值化
    threshold, final_image = cv2.threshold(output_image, 0, 255, cv2.THRESH_BINARY) # 二值化，阈值为0
    
    cv2.imwrite(out_dir + 'binary.png', final_image) # 保存图像
    


