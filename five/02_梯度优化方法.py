"""
动量法只从梯度入手  在一定程度上克服‘平缓’，‘鞍点’问题
AdaGrad,RMSProp 从学习率入手
Adam从两者都入手

学习率越小，梯度下降的速度越慢，
学习率越大，梯度下降的速度越快，导致震荡，甚至不收敛（梯度爆炸）

动量法公式（梯度的角度）：
St = β*St-1 + (1-β) * Gt
 St:本次指数移动加权平均结果
 β: 调节权重系数 越大数据与越平缓 本次梯度权重越小
 St-1:历史指数移动加权平均结果
 Gt:本次计算出的梯度(不考虑历史梯度)
 优点：快速收敛，减少梯度消失
 适用于非平稳优化问题
 
AdaGrad（学习率的角度）自适应：
   St = St-1 + Gt * Gt
    St:累计平方梯度
    St-1:历史累计平方梯度
    Gt:本次计算出的梯度(不考虑历史梯度)
梯度更新公式：
   学习率 = 学习率/(sqrt(St)+σ)
        σ：小常数1e-10,
梯度下降公式：
        W新 = W旧-调整后的学习率*Gt
通过对不同的参数分量使用不同的学习率，AdaGrad的学习率总体会逐渐减小
优点：自适应学习率
缺点：学习率过早地变低，较难找到最优解
适用于稀疏数据，如NLP或统计系统中的特征


RMSProp（学习率角度）:
  公式：
   St = β * St-1 + (1-β) * Gt * Gt
   β：调和权重技术
  学习率 = 学习率/(sqrt(St)+σ)
  优点：通过引入衰减系数β，控制历史梯度 对 历史梯度信息获取的多少
  适用于动态问题，非平稳目标函数，稀疏数据


Adam:自适应矩估计（学习率和梯度角度）
     公式：
     一阶矩估计:算均值 
            梯度：         Mt = β1 * Mt-1 + (1-β1) * Gt
            学习率：       St = β2 * St-1 + (1-β2) * Gt * Gt
     二阶矩估计：梯度的方差
                     Mt_hat = Mt/(1-β1^t)
                     St_hat = St/(1-β2^t)
    权重更新公式：
              W新 = W旧 -(学习率 / (sqrt(St_hat)+σ))* Mt_hat

    Adam = RMSprop + Momentum
     
     learning_rate = learning_rate/(sqrt(v_hat)+σ)
广泛适用于大量数据，各种深度学习问题，特别是非平稳和复杂问题

总结：Adam,SGD常用



人为：
学习率衰减优化方法


"""
import torch
import torch.nn as nn
#1.动量法 定义函数
def dm01():
    #1.初始化权重
    w = torch.tensor([1.0],requires_grad=True)
    #2.定义损失函数
    cri = ((w**2)/2.0).sum()
    #3.创建优化器 参2：学习率 momentum=0时：只考虑本次
    opt = torch.optim.SGD([w],lr=0.01,momentum=0.9)
    #4.计算梯度值：梯度清零+反向传播+参数更新
    opt.zero_grad()
    cri.sum().backward()
    opt.step()
    print(f"w:{w},w.grad:{w.grad}")
    #5.重复步骤，第二次更新权重
    cri = ((w**2)/2.0)
    opt.zero_grad()
    cri.sum().backward()
    opt.step()
    print(f"w:{w},w.grad:{w.grad}")

#2.Adagrad自适应方法 定义函数
def dm02():
    #1.初始化权重
    w = torch.tensor([1.0],requires_grad=True)
    #2.定义损失函数
    cri = ((w**2)/2.0).sum()
    #3.创建优化器,基于Adagrad
    opt = torch.optim.Adagrad([w],lr=0.01)
    #4.计算梯度值：梯度清零+反向传播+参数更新
    opt.zero_grad()
    cri.sum().backward()
    opt.step()
    print(f"w:{w},w.grad:{w.grad}")
    #5.重复步骤，第二次更新权重
    cri = ((w**2)/2.0)
    opt.zero_grad()
    cri.sum().backward()
    opt.step()
    print(f"w:{w},w.grad:{w.grad}")

#3.RMSProp自适应方法 定义函数
def dm03():
    #1.初始化权重
    w = torch.tensor([1.0],requires_grad=True)
    #2.定义损失函数
    cri = ((w**2)/2.0).sum()
    #3.创建优化器,基于Adagrad,alpha就是β
    opt = torch.optim.RMSprop([w],lr=0.01,alpha=0.99)
    #4.计算梯度值：梯度清零+反向传播+参数更新
    opt.zero_grad()
    cri.sum().backward()
    opt.step()
    print(f"w:{w},w.grad:{w.grad}")
    #5.重复步骤，第二次更新权重
    cri = ((w**2)/2.0)
    opt.zero_grad()
    cri.sum().backward()
    opt.step()
    print(f"w:{w},w.grad:{w.grad}")

def dm04():
    w = torch.tensor([1.0],requires_grad=True)
    cri = ((w**2)/2.0)
    opt = torch.optim.Adam([w],lr=0.01,betas=(0.9,0.999))
    #梯度清零 反向传播 更新参数
    opt.zero_grad()
    cri.sum().backward()
    opt.step()
    print(f"w:{w},w.grad:{w.grad}")
    cri = ((w**2)/2.0) 
    opt.zero_grad()
    cri.sum().backward()
    opt.step()  
    print(f"w:{w},w.grad:{w.grad}")





if __name__ == "__main__":
    #dm01()
    #dm02()
    #dm03()
    dm04()
