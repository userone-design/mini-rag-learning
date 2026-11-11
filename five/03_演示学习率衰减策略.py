"""
可以通过等间隔、指定间隔、指数方式 来手动控制学习率的调整
等间隔学习率：
  step_size=间隔的轮数 
  gamma=学习率缩放比例




  
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
#参2：间隔轮数 参3：学习率缩放比例
#1.等间隔  StepLR   lr = lr旧的学习率 * gamma 
#直观 适用于大批量数据，较为简单的任务
def dm01():
    #1.定义变量 初始学习率 训练轮数 每轮训练批次数
    lr = 0.1
    liter = 10
    epochs = 200
    #2.创建数据集 y_true x,w
    y_true = torch.tensor([0])
    x = torch.tensor([1.0],dtype=torch.float32)
    w = torch.tensor([1.0],requires_grad=True,dtype=torch.float32)
    #3.创建优化器 动量法  参1：待优化的参数 参2：学习率 参3：动量
    opt = torch.optim.SGD([w],lr=lr,momentum=0.9)
    #4.创建学习率衰减对象
    lr_schedule = torch.optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.5)
    #5.创建两个列表，分别表示：训练轮数，每轮训练的学习率
    lr_list,epoch_list = [],[]
    #6.循环训练过程
    for epoch in range(epochs):
        #7.获取当前轮数和学习率，并保存到列表中
        epoch_list.append(epoch)
        lr_list.append(lr_schedule.get_last_lr())#获取最后的lr
        #8.循环遍历
        for batch in range(liter):
            #9.先计算预测值，再计算损失
            y_pred = w * x
            #10.计算损失
            loss =(y_pred - y_true)**2
            opt.zero_grad()
            loss.backward()
            opt.step()
        #11.更新学习率
        lr_schedule.step()
    #12.保存结果
    print(f'lr_list{lr_list}')
    plt.plot(epoch_list,lr_list)
    plt.xlabel('epoch')
    plt.ylabel('lr')
    plt.title('学习率变化曲线')
    plt.legend()
    plt.show()

#2.指定间隔  MultiStepLR   lr = lr旧*gamma
#稳定训练 有时衰减过快
def dm02():
    #1.定义变量 初始学习率 训练轮数 每轮训练批次数
    lr = 0.1
    liter = 10
    epochs = 200
    #2.创建数据集 y_true x,w
    y_true = torch.tensor([0])
    x = torch.tensor([1.0],dtype=torch.float32)
    w = torch.tensor([1.0],requires_grad=True,dtype=torch.float32)
    #3.创建优化器 动量法  参1：待优化的参数 参2：学习率 参3：动量
    opt = torch.optim.SGD([w],lr=lr,momentum=0.9)
    #4.创建学习率衰减对象
    lr_schedule = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[50,125,160], gamma=0.5)
    #5.创建两个列表，分别表示：训练轮数，每轮训练的学习率
    lr_list,epoch_list = [],[]
    #6.循环训练过程
    for epoch in range(epochs):
        #7.获取当前轮数和学习率，并保存到列表中
        epoch_list.append(epoch) 
        lr_list.append(lr_schedule.get_last_lr())#获取最后的lr
        #8.循环遍历
        for batch in range(liter):
            #9.先计算预测值，再计算损失
            y_pred = w * x
            #10.计算损失
            loss =(y_pred - y_true)**2
            opt.zero_grad()
            loss.backward()
            opt.step()
        #11.更新学习率
        lr_schedule.step()
    #12.保存结果
    print(f'lr_list{lr_list}')
    plt.plot(epoch_list,lr_list)
    plt.xlabel('epoch')
    plt.ylabel('lr')
    plt.title('学习率变化曲线')
    plt.show()


#指数衰减  ExponentialLR  lr = lr * gamma ** epoch
#前期快，中后期慢，后期极慢
#平滑 收敛稳定性强 高精度
def dm03():
    #1.定义变量 初始学习率 训练轮数 每轮训练批次数
    lr = 0.1
    liter = 10
    epochs = 200
    #2.创建数据集 y_true x,w
    y_true = torch.tensor([0])
    x = torch.tensor([1.0],dtype=torch.float32)
    w = torch.tensor([1.0],requires_grad=True,dtype=torch.float32)
    #3.创建优化器 动量法  参1：待优化的参数 参2：学习率 参3：动量
    opt = torch.optim.SGD([w],lr=lr,momentum=0.9)

    #4.创建学习率衰减对象
    lr_schedule = torch.optim.lr_scheduler.ExponentialLR(opt,gamma=0.95)
    #5.创建两个列表，分别表示：训练轮数，每轮训练的学习率
    lr_list,epoch_list = [],[]
    #6.循环训练过程
    for epoch in range(epochs):
        #7.获取当前轮数和学习率，并保存到列表中
        epoch_list.append(epoch) 
        lr_list.append(lr_schedule.get_last_lr())#获取最后的lr
        #8.循环遍历
        for batch in range(liter):
            #9.先计算预测值，再计算损失
            y_pred = w * x
            #10.计算损失
            loss =(y_pred - y_true)**2
            opt.zero_grad()
            loss.backward()
            opt.step()
        #11.更新学习率
        lr_schedule.step()
    #12.保存结果
    print(f'lr_list{lr_list}')
    plt.plot(epoch_list,lr_list)
    plt.xlabel('epoch')
    plt.ylabel('lr')
    plt.title('学习率变化曲线')
    plt.show()








if __name__ == '__main__':
    #dm01()
    #dm02()
    dm03()