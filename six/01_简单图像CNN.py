"""
演示基础的图像操作：
    二值图：0-1  1通道
    灰度图：0-255  1通道
    索引图：0-255  1通道
    RGB图：0-255


    imshow
    imread
    imsave
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
\
def dm01():
    #1.定义全黑的图像
    #H 高度 W宽度 C通道
    img = np.zeros((200,200,3))

    #绘制图片
    plt.imshow(img)
    plt.axis('off')#把坐标系关闭
    plt.show()

  #2.定义全白的图像
    #H 高度 W宽度 C通道
    img = np.ones((200,200,3))
    print(f"img:{img}")
    #绘制图片
    plt.imshow(img)
    plt.show()

def dm02():
    img1 = plt.imread("./data/img.jpg")
    print(f"img1:{img1}")
    print(f"img1.shape:{img1.shape}")
    #保存
    #plt.imsave("./data/img1_copy.jpg",img1)











if __name__ == '__main__':
    #dm01()
    dm02()