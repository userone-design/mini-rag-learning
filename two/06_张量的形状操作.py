"""
reshape() squeze() unsqueze() transpose() permute() 交换维度
 view()   不能做不连续的张量
 contigous() 把不连续的张量转化为连续的张量
 is_contigous()  判断是否连续
"""
import torch
torch.manual_seed(24)
def dm01():
    #reshape() 不改变数据，只改变形状
    t1 = torch.randint(1,10,(2,3))
    print(f"t1:{t1},shape:{t1.shape},row:{t1.shape[0]},colums:{t1.shape[1],t1.shape[-1]}")
    #reshape()把t1转为1行6列
    t2 = t1.reshape(3,2)
    print(f"t2:{t2},shape:{t2.shape},row:{t2.shape[0]},colums:{t2.shape[1],t2.shape[-1]}")

def dm02():
    #unsqueeze() 在指定的轴上增加一个（1）维度，等价于 升维
    t1 = torch.randint(1,10,(2,3))
    print(f"t1:{t1},shape:{t1.shape}")
    #在0维上添加一个维度
    t2 = t1.unsqueeze(0)
    print(f"t2:{t2},shape:{t2.shape}")
    #在1维上添加一个维度
    t3 = t1.unsqueeze(1)
    print(f"t3:{t3},shape:{t3.shape}")
    #在2维上添加一个维度
    t4 = t1.unsqueeze(2)
    print(f"t4:{t4},shape:{t4.shape}")

    #删除所有维1的维度
    t5 = torch.randint(1,10,size=(2,1,3,1,1))
    print(f"t5:{t5},shape:{t5.shape}")
    t6 = torch.squeeze()
    print(f"t6:{t6},shape:{t6.shape}")


#permute()可以任意换两个维度的位置
def dm03():
    t1 = torch.randint(1,10,size=(2,3,4))
    print(f"t1:{t1},shape:{t1.shape}")
    print("-"*30)
    t2 = t1.transpose(0,1)#第0维和第1维做交换 
    print(f"t2:{t2},shape:{t2.shape}")
    print("-"*30)
    t3 = t1.transpose(0,2)#第0维和第2维做交换 
    print(f"t3:{t3},shape:{t3.shape}")
    t4 = t1.permute(4,2,3)
    print(f"t4:{t4},shape:{t4.shape}")

#view只能用于修改连续的张量，经过transport，permute函数的处理之后无法用view
#
def dm04():
    t1 = torch.randint(1,10,size=(2,3))
    print(t1.is_contiguous())
    print(f"t1:{t1},shape:{t1.shape}")

#   2行3列->3行2列
    t2 = t1.view(3,2)
    print(t2.is_contiguous())
    print(f"t2:{t2},shape:{t2.shape}")

    t3 = t1.transpose(0,1)
    print(t3.is_contiguous())
    print(f"t3:{t3},shape:{t3.shape}")
#   通过contigous转化连续
    t4 = t3.contiguous().view(2,3)
    print(t4.is_contiguous())









if __name__ == '__main__':
 #   dm01()
 #   dm02()
 #   dm03()
    dm04()







