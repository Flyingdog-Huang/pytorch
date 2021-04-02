# 1.开始用torch.nn包里面的函数搭建网络
# 2.模型保存为pt文件与加载调用
# 3.Torchvision.transofrms来做数据预处理
# 4.DataLoader简单调用处理数据集

from numpy.core.numeric import correlate
import torch as t
from torch.utils.data import DataLoader
import torchvision as tv

#预处理数据
transfrom = tv.transforms.Compose([
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.5, ), (0.5, )),
])
# Totensor表示把灰度图像素值从0~255转化为0~1之间
# Normalize表示对输入的减去0.5, 除以0.5

# 数据集
train_ts = tv.datasets.MNIST(root='./data',
                             train=True,
                             download=True,
                             transform=transfrom)
test_ts = tv.datasets.MNIST(root='./data',
                            train=False,
                            download=True,
                            transform=transfrom)
train_dl = DataLoader(train_ts, batch_size=32, shuffle=True, drop_last=False)
test_dl = DataLoader(test_ts, batch_size=64, shuffle=True, drop_last=False)

# 网络结构
# 输入层：784个神经元
# 隐藏层：100个神经元
# 输出层：10个神经元
model = t.nn.Sequential(t.nn.Linear(784, 100), t.nn.ReLU(),
                        t.nn.Linear(100, 10), t.nn.LogSoftmax(dim=1))

#定义损失函数与优化函数
loss_fn = t.nn.NLLLoss(reduction="mean")
optimizer = t.optim.Adam(model.parameters(), lr=1e-3)

# 开启训练
for s in range(5):
    print("run in step : %d" % s)
    for i, (x_train, y_train) in enumerate(train_dl):
        x_train = x_train.view(x_train.shape[0], -1)
        y_pred = model(x_train)
        train_loss = loss_fn(y_pred, y_train)
        if (i + 1) % 100 == 0:
            print(i + 1, train_loss.item())
        model.zero_grad()  # 每次训练结束后，梯度置零，避免累加带来的误差
        train_loss.backward()
        optimizer.step()

# 测试模型准确率
total = 0
correct_count = 0
for test_images, test_labels in test_dl:
    for i in range(len(test_labels)):
        image = test_images[i].view(1, 784)
        with t.no_grad():
            pred_labels = model(image)
        plabels = t.exp(pred_labels)
        probs = list(plabels.numpy()[0])
        pred_label = probs.index(max(probs))
        true_label = test_labels.numpy()[i]
        if true_label == pred_label:
            correct_count += 1
        total += 1

# 打印准确率和保存模型
print("total acc : %.2f\n" % (correct_count / total))
t.save(model, './nn_mnist_model.pt')
