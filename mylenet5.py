import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utils import *

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = BackBone()
        self.head = Head()
    def forward(self, x):
        x = self.backbone.forward(x)
        x = self.head.forward(x)
        return x


class BackBone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=0)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5,padding=0)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x

class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    batch_size = 64
    set_seed(42)
    # 数据处理部分
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5,))
    ])

    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    # 模型搭建
    model = LeNet().to(device=device)
    from torchsummary import summary
    # 增加 device=device.type 参数
    print(summary(model=model, input_size=(1,28,28), 
                  batch_size=batch_size, device=device.type))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练
    epochs = 10
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )
        val_loss, val_acc = evaluate(
            model,
            test_loader,
            criterion,
            device
        )
        print(train_loss)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
    # 可视化
    plot_curves(
        train_losses,
        val_losses,
        train_accs,
        val_accs
    )
    plt.savefig("lenet5_result.png")
    plt.close()

    # 保存模型
    save_model(model)

