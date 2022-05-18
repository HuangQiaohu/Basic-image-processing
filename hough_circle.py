# -*- coding: utf-8 -*-


import cv2 # 利用opencv进行图像处理
from hough import Hough_transform # 从霍夫变化类中引入霍夫变换函数
from canny_edge import get_gradient_and_direction, NMS, double_threshold # 从任务一引入canny边缘检测函数

if __name__ == '__main__':
    input_dir = './img/' # 输入图片文件夹
    out_dir = './task2/' # 输出图片文件夹
    
    image = cv2.imread(input_dir + 'wheel.png', 0) # 读取灰度图像
    img_RGB = cv2.imread(input_dir + 'wheel.png') # 读取原彩色图像

    # canny边缘检测   
    # 第一步高斯滤波
    smoothed_image = cv2.GaussianBlur(image, (5,5), 4) # 高斯滤波核尺寸为3*3 
    # 第二步求图像梯度
    gradients, direction = get_gradient_and_direction(smoothed_image)
    # 第三步非极大值抑制
    nms = NMS(gradients, direction)
    #第四步双阈值处理和滞后边界跟踪
    output_image = double_threshold(nms, 60, 140)
    #第五步二值化
    threshold, final_image = cv2.threshold(output_image, 0, 255, cv2.THRESH_BINARY)   
    cv2.imwrite(out_dir + "canny_binary.jpg", final_image) # 储存梯度二值化图像

    # hough变换圆检测    
    Hough_transform_threshold = 64 # 霍夫变换阈值
    Hough = Hough_transform(final_image, direction, Hough_transform_threshold) # 霍夫变换圆检测
    
    vote = Hough.Hough_transform_algorithm() # 投票矩阵    
    heatmap = cv2.normalize(vote, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U) # 归一化
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET) # 热图
    cv2.imwrite(out_dir + "heatmap.jpg", heatmap)# 保存热图

    r = Hough.Select_Circle() # 检测出的圆半径
    for circle in Hough.circles: # 遍历每个检测出来的圆
        cv2.circle(img_RGB, (int(circle[0]), int(circle[1])), r, (0,0,255), 2) # 在原图上画圆，即标出被检测出来的车轮   
    cv2.imwrite(out_dir + "hough_result.jpg", img_RGB) # 保存图像





