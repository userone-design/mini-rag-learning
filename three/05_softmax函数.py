"""
将多分类的结果以概率的形式展现,概率和相加为1 
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

# scores = torch.tensor([0.2,0.02,0.15,0.15,1.3,0.5,0.06,1.1,0.05,3.75])
scores = torch.tensor([[0.2,0.35,0.1,0.46],[0.1,0.13,0.05,2.79]])
pro = torch.softmax(scores,dim=0)
print(pro)