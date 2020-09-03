import torch
import numpy as np

print(torch.__version__)

#定义矩阵
x = torch.empty(2, 2) #初始化为0
print(x)

#定义随机初始化矩阵
x = torch.randn(2, 2)
print(x)

#定义初始化为0
x = torch.zeros(3, 3)
print(x)

#定义数据为tensor
x = torch.tensor([5.1, 2., 3., 1.])
print(x)

#操作
a = torch.tensor([1., 2., 3., 4., 5., 6., 7., 8.])
b = torch.tensor([11., 12., 13., 14., 15., 16., 17., 18.])
c = a.add(b) #矩阵相加
print(c)

#维度变换 2X4
a = a.view(-1, 4)
b = b.view(-1, 4)
c = torch.add(a, b) #矩阵相加
print(c, a.size(), b.size())

#torch to numpy and visa
na = a.numpy()
nb = b.numpy()
print("\na =", na, "\nb =", nb)

# 操作
d = np.array([21., 22., 23., 24., 25., 26., 27., 28.], dtype=np.float32)
print(d.reshape(2, 4))
d = torch.from_numpy(d.reshape(2, 4)) #将数组转换成张量且共享内存
sub = torch.sub(c, d) #矩阵相减
print(sub, "\n sub = ", sub.size())

# using CUDA
if torch.cuda.is_available():
    result = d.cuda() + c.cuda()
    print("\n result = ", result)

# 自动梯度
x = torch.randn(1, 5, requires_grad=True) # allows for fine grained exclusion of subgraphs from gradient computation and can increase efficiency.
y = torch.randn(5, 3, requires_grad=True) # 允许从梯度计算中细粒度地排除子图，可以提高效率。
z = torch.randn(3, 1, requires_grad=True)
print("\n x = ", x, "\n y = ", y, "\n z = ", z)
xy = torch.matmul(x, y) #矩阵乘法
xyz = torch.matmul(xy, z)
xyz.backward() #反向传播
print(x.grad, y.grad, z.grad)