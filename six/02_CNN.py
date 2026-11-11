"""
CNN: 卷积神经网络 
三部分：卷积层、池化层、全连接层
1卷积层.线性乘积求和，提取 局部特征    激励层 ：激活函数输入数据转化为输出数据
2池化层.用来大幅度降低参数量级 降维
3全连接层.全连接层用来输出想要的结果
卷积层的作用：
    1.特征提取
    2.共享权重，卷积核在整个输入图像上共用同一个
    3.局部连接
    4.空间不变性
池化层作用：不会改变通道数   输入 = 输出
    1.降低参数量级
    2.降低计算复杂度
    3.降维
    nn.MaxPool2d(kernel_size=2,stride=2,padding=0) 最大池化
    nn.AvgPool2d(kernel_size=2,stride=2,padding=0) 平均池化


filter: 卷积核,滤波器 == 神经元数量
卷积运算：
    本质上就是卷积核与输入做点积
    输入：[N,C,H,W]  N:批次数  C:通道数  H:高  W:宽

Padding(填充) 用于处理卷积时图像边缘的像素丢失问题
              目的是在输入图像周围添加一圈像素0
    主要作用：
             1.保持输入输出的尺寸一致
             2.保留边缘信息
             3.避免图像快速缩小
    类型：
        1.SAME: 填充足够的0，保证输出尺寸和输入尺寸一致
        2.VALID: 不填充，输出尺寸会缩小

        
Stride(步长) 卷积核移动的步长
       作用：
            1.降低计算复杂度：大步长减少计算量
            2.增大感受野
        通常设置为1   

多通道卷积核   三通道的先对应通道做点乘，然后做加法

多卷积核卷积运算 

特征图大小计算：
          输出尺寸 = (输入特征图大小 + 2*填充 - 卷积核大小) / 步长 + 1



卷积层的API       

"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

#1.定义函数用于完成图像的加载，卷积，特征图可视化操作
def dm01():
    img = plt.imread("./data/img.jpg")
    #print(f"img:{img},shape:{img.shape}")

    #把图像的形状从 H W C 改为 C H W img->张量->转换维度
    img2 = torch.tensor(img,dtype=torch.float).permute(2,0,1)
    #print(f"img2:{img2},shape:{img2.shape}")

    #再增加一个维度 从C H W 改为 N(1) C H W 一张3通道的640*640像素的图
    img3 = img2.unsqueeze(0) 
    print(f"img3:{img3},shape:{img3.shape}")

    #创建卷积层对象
    conv = nn.Conv2d(in_channels=3,out_channels=4,kernel_size=3,stride=2,padding=0)

    #卷积层进行运算
    conv_img = conv(img3)
    print(f"conv_img:{conv_img},shape:{conv_img.shape}")

    #可视化
    img4 = conv_img[0]
    print(f"img4:{img4},shape:{img4.shape}") #[4,319,319]
    #chw -> hwc
    img5 = img4.permute(1,2,0)
    print(f"img5:{img5},shape:{img5.shape}")
    feature1 = img5[:,:,1].detach().numpy() #第0通道 (319,319)像素图
    plt.imshow(feature1)
    plt.show()



if __name__ == "__main__":
    dm01()