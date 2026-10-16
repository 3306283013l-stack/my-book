import torch
import torch.nn as nn
from torch.nn import Sequential
from matplotlib import pyplot as plt
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torchvision.transforms as transforms

epochs = 10
batch_size = 64
lr = 0.001

# 转换导集

# 设置数据转换方式
transform = transforms.Compose([
    transforms.ToTensor(),  # 把数据转换为张量（Tensor）
    transforms.Normalize(  # 标准化，即使数据服从期望值为 0，标准差为 1 的正态分布
        mean=[0.5, ],  # 期望
        std=[0.5, ]  # 标准差
    )
])

# 训练集导入
data_train = datasets.MNIST(root='data/', transform=transform, train=True, download=True)
# 数据集导入
data_test = datasets.MNIST(root='data/', transform=transform, train=False)

# 数据装载

# 训练集装载
dataloader_train = DataLoader(dataset=data_train, batch_size=64, shuffle=True)
# 数据集装载
dataloader_test = DataLoader(dataset=data_test, batch_size=64, shuffle=True)

# 数据预览
images, labels = next(iter(dataloader_train))
img = make_grid(images)
img = img.numpy().transpose(1, 2, 0)
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]
img = img * std + mean
print([labels[i] for i in range(16)])
plt.imshow(img)
plt.show()


# 构建卷积神经网络
class CNN(nn.Module):  # 从父类 nn.Module 继承
    def __init__(self):  # 相当于 C++ 的构造函数
        # super() 函数是用于调用父类(超类)的一个方法，是用来解决多重继承问题的
        super(CNN, self).__init__()

        # 第一层卷积层。Sequential(意为序列) 括号内表示要进行的操作
        self.conv1 = Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 第二卷积层
        self.conv2 = Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 全连接层（Dense，密集连接层）
        self.dense = Sequential(
            nn.Linear(7 * 7 * 128, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 10)
        )

    def forward(self, x):  # 正向传播
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x = x2.view(-1, 7 * 7 * 128)
        x = self.dense(x)
        return x


# 训练和参数优化

# 定义求导函数
def get_Variable(x):
    x = torch.autograd.Variable(x)  # Pytorch 的自动求导

    # 判断是否有可用的 GPU
    return x.cuda() if torch.cuda.is_available() else x


# 定义网络
cnn = CNN()

# 判断是否有可用的 GPU 以加速训练
if torch.cuda.is_available():
    cnn = cnn.cuda()

# 设置损失函数为 CrossEntropyLoss（交叉熵损失函数）
loss_F = nn.CrossEntropyLoss()

# 设置优化器为 Adam 优化器
optimizer = torch.optim.Adam(cnn.parameters(), lr=lr)

# 训练
for epoch in range(epochs):
    running_loss = 0.0  # 一个 epoch 的损失
    running_correct = 0.0  # 准确率
    print("Epoch [{}/{}]".format(epoch, epochs))
    for data in dataloader_train:
        # DataLoader 返回值是一个 batch 内的图像和对应的 label
        X_train, y_train = data
        X_train, y_train = get_Variable(X_train), get_Variable(y_train)
        outputs = cnn(X_train)
        _, pred = torch.max(outputs.data, 1)
        # 后面的参数代表降低 outputs.data 的维度 1 个维度再输出
        # 第一个返回值是张量中最大值，第二个返回值是最大值索引
        # -------------------下面内容与随机梯度下降类似-----------------------------
        optimizer.zero_grad()
        # 梯度置零
        loss = loss_F(outputs, y_train)
        # 求损失
        loss.backward()
        # 反向传播
        optimizer.step()
        # 更新所有梯度
        # --------------------上面内容与随机梯度下降类似----------------------------
        running_loss += loss.item()  # 此处 item() 表示返回每次的 loss 值
        running_correct += torch.sum(pred == y_train.data)

    testing_correct = 0.0

    for data in dataloader_test:
        X_test, y_test = data
        X_test, y_test = get_Variable(X_test), get_Variable(y_test)
        outputs = cnn(X_test)
        _, pred = torch.max(outputs, 1)
        testing_correct += torch.sum(pred == y_test.data)
        # print(testing_correct)
    print("Loss: {:.4f}  Train Accuracy: {:.4f}%  Test Accuracy: {:.4f}%".format(
        running_loss / len(data_train), 100 * running_correct / len(data_train),
        100 * testing_correct / len(data_test)))

# 保存模型
torch.save(cnn, 'data/model.pth')
