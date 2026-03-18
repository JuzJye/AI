import torch
import torch.nn as nn
from torchvision import datasets, transforms
from pathlib import Path
import os
from torch.utils.data import DataLoader
from torchsummary import summary
from tqdm import tqdm
import matplotlib.pyplot as plt

class Conv(nn.Module):
    def __init__(self, in_c, out_c, kernel, stride=1, padding=1, is_act=True):
        
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel, stride, padding)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU()
        self.is_act = is_act

    def forward(self, x):
        x = self.bn(self.conv(x))   # 卷积后进行归一化
        return self.act(x) if self.is_act else x
    
class MaxPooling(nn.Module):
    def __init__(self, kernel, stride, padding=0) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=kernel, stride=stride, padding=padding)

    def forward(self, x):
        return self.pool(x)

class FC(nn.Module):
    def __init__(self, in_c, out_c, is_act=True) -> None:
        super().__init__()
        self.fc1= nn.Linear(in_c, out_c)
        self.act = nn.ReLU()
        self.is_act = is_act
    def forward(self, x):
        x = self.fc1(x)
        return self.act(x) if self.is_act else x

class Vgg16(nn.Module):
    def __init__(self, in_c=3, class_num=5) -> None:
        '''
        in_c: 输入通道3的图片
        class_num: 输出5个类别
        '''
        super().__init__()
        self.class_num = class_num
        # 搭模型结构
        self.conv1_1 = Conv(in_c=in_c,out_c=64,kernel=3,is_act=True)
        self.conv1_2 = Conv(in_c=64,out_c=64,kernel=3,is_act=True)
        self.maxpool1 = MaxPooling(2,2)
     
        self.conv2_1 = Conv(in_c=64,out_c=128,kernel=3,is_act=True)
        self.conv2_2 = Conv(in_c=128,out_c=128,kernel=3,is_act=True)
        self.maxpool2 = MaxPooling(2,2)

        self.conv3_1 = Conv(in_c=128,out_c=256,kernel=3,is_act=True)
        self.conv3_2 = Conv(in_c=256,out_c=256,kernel=3,is_act=True)
        self.conv3_3 = Conv(in_c=256,out_c=256,kernel=3,is_act=True)
        self.maxpool3 = MaxPooling(2,2)

        self.conv4_1 = Conv(in_c=256,out_c=512,kernel=3,is_act=True)
        self.conv4_2 = Conv(in_c=512,out_c=512,kernel=3,is_act=True)
        self.conv4_3 = Conv(in_c=512,out_c=512,kernel=3,is_act=True)
        self.maxpool4 = MaxPooling(2,2)

        self.conv5_1 = Conv(in_c=512,out_c=512,kernel=3,is_act=True)
        self.conv5_2 = Conv(in_c=512,out_c=512,kernel=3,is_act=True)
        self.conv5_3 = Conv(in_c=512,out_c=512,kernel=3,is_act=True)
        self.maxpool5 = MaxPooling(2,2)
        
        self.flatten = nn.Flatten()
        self.fc1 = FC(7*7*512, 128, is_act=True)
        self.fc2 = FC(128, 128, is_act=True)
        self.fc3 = FC(128, class_num, is_act=False)


    def forward(self, x):
        x = self.maxpool1(self.conv1_2(self.conv1_1(x)))
        x = self.maxpool2(self.conv2_2(self.conv2_1(x)))
        x = self.maxpool3(self.conv3_3(self.conv3_2(self.conv3_1(x))))
        x = self.maxpool4(self.conv4_3(self.conv4_2(self.conv4_1(x))))
        x = self.maxpool5(self.conv5_3(self.conv5_2(self.conv5_1(x))))
        x = self.flatten(x)
        x = self.fc3(self.fc2(self.fc1(x)))
        return x
    

if __name__ == "__main__":
    # 测试模型搭建是否成功
    # x = torch.randn(1,3,224,224)
    # net = Vgg16(3, 5)
    # pre_y = net(x)
    
    # 参数定义
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    batch_size = 8
    num_classes = 5
    epochs = 35
    init_lr = 1e-5
    current_lr = init_lr
    isLoad = False
    isWeight = False
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    best_val_accuracy = 0.0
    best_val_loss = float('inf')
    train_path = Path("../flower_photos/train") # 为了健壮性这么写 不然macOS直接写路径就行
    test_path = Path("../flower_photos/test")
    print("device:",device)
    print("参数设置完毕，训练开始^^")

    # 数据处理部分
    my_transform = transforms.Compose([        
        # 调整图片尺寸
        transforms.Resize((224,224)),
        # 以下为数据增强：
        # 随机均衡化
        transforms.RandomEqualize(p=0.5),
        # 随机水平翻转
        transforms.RandomHorizontalFlip(p=0.1),
        # 随机旋转15度
        transforms.RandomRotation(15),
        # 随机垂直翻转
        transforms.RandomVerticalFlip(p=0.1),
        # 按需增强
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        # 高斯模糊
        transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2)),
        # 转化为tensor
        transforms.ToTensor(),
        # 标准化 这里的数字如何确定？
        transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
    ])

    # 准备数据集
    train_dataset = datasets.ImageFolder(
        root=train_path,
        transform=my_transform
    )

    val_dataset = datasets.ImageFolder(
        root=test_path,
        transform=my_transform
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # 开始训练
    model = Vgg16(in_c=3, class_num=5).to(device=device)
    # print(summary(model=model,input_size=(3,224,224),
    #               batch_size=batch_size,device=device.type))
    # 加载训练中断后保存的模型状态（即“断点续训”）或者上次训练得到的最好指标权重(即“增量训练”)
    if isLoad and os.path.exists('./best.pt'):
        print("加载初始权重best.pt")
        checkpoint = torch.load('best.pt')
        # 从字典 checkpoint 中提取出模型的参数（权重和偏置）
        # "model_state_dict": "权重",
        # "current_lr": "上次结束的时候的学习率",
        # "best_val_accuracy": "上次结束时最佳acc",
        # "best_val_loss": "上次结束的损失"
        model.load_state_dict(checkpoint['model_state_dict'])
        current_lr = checkpoint['current_lr']
        best_val_accuracy = checkpoint['best_val_accuracy']
        best_val_loss = checkpoint['best_val_loss']
        print(f'历史学习率={current_lr},上次验证损失={best_val_loss}, 验证acc={best_val_accuracy}')
        init_lr = current_lr
    # 根据不同类别的训练数据个数，给一个权重系数（即”类别平衡“）
    # 多的数据 系数小一点，少的数据 系数大一点 
    if isWeight:
        class_weights = torch.tensor([0.5, 1.0, 1.0, 1.0, 1.0]).float()
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device=device))
    else:
        criterion = nn.CrossEntropyLoss()

    # 设置优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)

    for epoch in range(epochs):
        model.train()
        train_loss_epoch=0.0
        train_acc_epoch=0.0
        total_train_samples=0
        train_class_correct=[0]* num_classes
        train_class_total=[0]* num_classes

        for input_x,label in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}/{epochs}",unit="batch"):
            optimizer.zero_grad()
            input_x = input_x.to(device)
            label = label.to(device)
            pred_y = model(input_x)
            loss = criterion(pred_y, label)
            loss.backward()
            optimizer.step()
            pred = torch.argmax(pred_y, dim=1)
            accuracy = torch.sum(pred == label)/label.size(0)
            # 整体指标
            train_loss_epoch += loss.item() * label.size(0)
            train_acc_epoch += accuracy.item() * label.size(0)
            total_train_samples += label.size(0)
            # 统计每个类别的准确率
            for i in range(num_classes):
                train_class_correct[i] += ((pred == i) & (label == i)).sum().item()
                train_class_total[i] += (label == i).sum().item()
        avg_train_loss = train_loss_epoch / total_train_samples
        avg_train_acc = train_acc_epoch / total_train_samples
        train_losses.append(avg_train_loss)
        train_accs.append(avg_train_acc)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_train_loss:.4f}, Acc: {avg_train_acc:.2%}")
        print("Class-wise Training Accuracy: ")
        for i in range(num_classes):
            print(f"Class {i}: {train_class_correct[i] / train_class_total[i]:.2%}")

        # 验证集
        model.eval()
        val_loss_epoch=0.0
        val_acc_epoch=0.0
        total_val_samples=0
        val_class_correct=[0]*num_classes
        val_class_total=[0]*num_classes
        
        with torch.no_grad():
            for input_x,label in tqdm(val_dataloader, desc=f"Validation Epoch {epoch+1}/{epochs}",unit="batch"):
                input_x = input_x.to(device)
                label = label.to(device)
                pred_y = model(input_x)
                loss = criterion(pred_y, label)
                pred = torch.argmax(pred_y, dim=1)
                accuracy = torch.sum(pred == label)/label.size(0)
                # 整体指标
                val_loss_epoch += loss.item() * label.size(0)
                val_acc_epoch += accuracy.item() * label.size(0)
                total_val_samples += label.size(0)
                # 统计每个类别的准确率
                for i in range(num_classes):
                    val_class_correct[i] += ((pred == i) & (label == i)).sum().item()
                    val_class_total[i] += (label == i).sum().item()
        avg_val_loss = val_loss_epoch / total_val_samples
        avg_val_acc = val_acc_epoch / total_val_samples
        val_losses.append(avg_val_loss)
        val_accs.append(avg_val_acc)
        print(f"Validation Loss: {avg_val_loss:.4f}, Validation Acc: {avg_val_acc:.2%}")
        print("Class-wise Validation Accuracy: ")
        for i in range(num_classes):
            print(f"Class {i}: {val_class_correct[i] / val_class_total[i]:.2%}")

        # 保存模型
        if avg_val_acc > best_val_accuracy:
            best_val_accuracy = avg_val_acc
            best_val_loss = avg_val_loss
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'current_lr': current_lr,
                'best_val_accuracy': best_val_accuracy,
                'best_val_loss': best_val_loss
            }
            torch.save(checkpoint, 'best.pt')
            print(f"Best validation accuracy updated: {best_val_accuracy:.2%}, best validation loss: {best_val_loss:.4f}")
    # 可视化
    # 绘制训练和验证损失曲线
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.plot(range(1,epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1,epochs+1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # 绘制训练和验证准确度曲线
    plt.subplot(1,2,2)
    plt.plot(range(1,epochs+1), train_accs, label='Train Accuracy')
    plt.plot(range(1,epochs+1), val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()
    plt.savefig('vgg16_result.png')
    plt.close()