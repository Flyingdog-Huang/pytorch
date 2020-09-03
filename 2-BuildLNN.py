# 1.开始用torch.nn包里面的函数搭建网络
# 2.模型保存为pt文件与加载调用
# 3.Torchvision.transofrms来做数据预处理
# 4.DataLoader简单调用处理数据集

import torch as t
from torch.utils.data import DataLoader
import torchvision as tv

# 数据集
train_ts = tv.datasets.MNIST(root='./data')

#预处理数据
transfrom = tv.transforms.Compose([tv.transforms.ToTensor(),
                                   tv.transforms.Normalize((0.5,), (0.5,)), ])
# Totensor表示把灰度图像素值从0~255转化为0~1之间
# Normalize表示对输入的减去0.5, 除以0.5

# 网络结构
# 输入层：784个神经元
# 隐藏层：100个神经元
# 输出层：10个神经元
model = t.nn.Sequential(t.nn.Linear(784, 100),
                        t.nn.ReLU(),
                        t.nn.Linear(100, 10),
                        t.nn.LogSoftmax(dim=1))

#定义损失函数与优化函数
loss_fn = t.nn.NLLLoss(reduction="mean")
optimizer = t.optim.Adam(model.parameters(), lr=1e-3)

# 开启训练
for s in range(5):
    print("run in step : %d"%s)
    for i, (x_train, y_train) in enumerate(tra):