"""
优化器:optimizer
numpy->张量->数据集对象->数据加载器
""" 
import torch
from torch.utils.data import TensorDataset 
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

def create_datasert():
    #创建数据集
    x,y,coef = make_regression(
       n_samples= 100,#100个样本
        n_features = 1,
        noise = 10,#噪声越大，样本点越散
        bias=14.5,#b
        random_state=3,#随机种子，随机种子相同，输出数据相同
        coef=True# 权重，，是否返回系数，默认FALSE 
    )
    #封装成张量
    x = torch.tensor(x,dtype=torch.float)
    y = torch.tensor(y,dtype=torch.float)
    return x,y,coef

#2定义函数，表示模型函数
def train(x,y,coef):
    #1创建数据集对象 把tensor->数据集对象->数据加载器
    dataset = TensorDataset(x,y)
    #创建数据加载器对象 shuffle：是否打乱（训练集打乱，测试集不打乱 ）
    dataloader = DataLoader(dataset,batch_size=16,shuffle=True)
    #3创建初始线性回归模型 (1,1) 输入特征维度，输出特征维度
    model = nn.Linear(1,1)
    #创建损失函数对象
    criterion = nn.MSELoss()
    #创建优化器对象 模型参数，学习率
    optimizer = optim.SGD(model.parameters(),lr=0.01)

    #具体训练过程,训练轮数，每轮平均损失，训练总损失值，训练的样本数
    epochs,loss_list,total_loss,total_sample = 100,[],0.0,0
    for epochs in range(epochs):#epochs:0,1,2,......
        total_loss = 0.0
        total_sample = 0
        for train_x,train_y in dataloader:#100个样本每轮可被分成7批（16，16，16，16，16，16，4）
            #模型的预测
            y_pred= model(train_x)
            #计算(每批的平均)损失
            loss = criterion(y_pred,train_y.reshape(-1,1))#n行1列
            #计算总损失，样本批次数
            total_loss += loss.item()
            total_sample +=1#每批的次数加1 
            #梯度清零 +反向传播+梯度更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()#梯度更新
        #把本轮的平均损失值添加到列表中
        loss_list.append(total_loss / total_sample)
        print(f"轮数：{epochs+1},平均损失值：{total_loss/total_sample}")
    print(f"100轮后平均损失：{loss_list}")
    print(f"模型参数，权重：{model.weight},偏置：{model.bias}")
    #绘制图像
    plt.plot(loss_list)
    plt.title('损失曲线变化图')
    plt.grid()
    plt.show()
    #绘制样本点分布情况
    plt.scatter(x,y)
    #绘制训练模型的预测值
    #x:100个样本点的特征
    y_pred = torch.tensor(data=[v*model.weight+model.bias for v in x])
    #y_pred = model(x).detach().numpy() 

    #v:它会把x里的每一个值挨个取出来，起名 v
    #然后计算 v * 权重+偏置，最后把算出来的100个结果打包成一个列表。
    y_true = torch.tensor(data=[v*coef+14.5 for v in x])
    #y_true = (x * coef + 14.5).detach().numpy()

    plt.plot(x,y_pred,color='red',label='预测值')
    plt.plot(x,y_true,color='green',label='真实值')
    plt.legend()
    plt.grid()
    plt.show()


    plt.show()


#b=14.5
#w=27.47085 运行之后的
#测试
if __name__ == '__main__':
   #创建数据集
   x,y,coef = create_datasert()
#  print(f"x:{x},y:{y},coef:{coef}")
#模型训练
   train(x,y,coef)








