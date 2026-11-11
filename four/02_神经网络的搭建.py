"""
_init_定义网络中的层结构 主要是全连接层并进行初始化
forward方法在实例化模型后，输入数据时会调用forward方法进行前向传播计算
神经网络搭建流程:
1.定义一个类,继承nn.Module
2.在_init_方法中定义网络层结构：父类成员初始化
3.在forward方法中完成前向传播计算

深度学习案例步骤：
1.数据集准备
2.搭建网络模型
3.模型训练，模型测试

"""
import torch
import torch.nn as nn
from torchsummary import summary
from torch.utils.data import TensorDataset 
from torch.utils.data import DataLoader
from torch import nn
from torch import optim

#1.搭建神经网络
class ModelDemo(nn.Module):
    #1.在init中完成初始化父类成员
    def __init__(self):
        super().__init__()
        #2.搭建神经网络 隐藏层+输出层
        self.linear1 = nn.Linear(3, 3) #输入层到隐藏层
        self.linear2 = nn.Linear(3, 2) #隐藏层到输出层
        #输出层
        self.out = nn.Linear(2, 2) #输出层

        #对隐藏层和输出层进行初始化
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)

        nn.init.kaiming_normal_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    #2.前向传播计算 输入层-》隐藏层-》输出层
    def forward(self, x):
        #第一层隐藏层计算：加权求和+激活函数
        # x = self.linear1(x)#加权求和
        # x = torch.sigmoid(x)#激活函数
        #也可以写成一行
        x = torch.sigmoid(self.linear1(x))
        #第二层隐藏层计算：加权求和+激活函数
        x = torch.relu(self.linear2(x))
        #输出层计算：加权求和+激活函数
        x = torch.softmax(self.out(x), dim=1) #dim=1表示按行计算
        return x    

#3.模型训练
def train():
    #1.创建模型
    my_model = ModelDemo()
    print(f"模型结构：{my_model}")
    #2.创建数据集样本
    data = torch.randn(5,3) #100个样本，每个样本3个特征
    print(f"数据集样本：{data},data.shape={data.shape}")
    print(f"data.requires_grad={data.requires_grad}")
    #3.调用神经网络 进行模型训练
    output = my_model(data)#底层会自动调用forward进行前向传播计算
    print(f"模型输出结果：{output},output.shape={output.shape}")
    print(f"output.requires_grad={output.requires_grad}")

    print("-"*30)
    #计算 查看 模型参数
    print("========计算模型参数========")
    #输出的output shape：-1代表不确定
    summary(my_model, (5,3)) #输入数据的特征维度为5行3列
    print("========查看模型参数========")
    for name, param in my_model.named_parameters():
        print(f"参数名称：{name},参数值：{param}\n,参数形状：{param.shape}")

    











if __name__ == '__main__':
    train()
