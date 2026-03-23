import torch
import torch.nn as nn
from torchvision import datasets, transforms
from pathlib import Path
import os
from torch.utils.data import DataLoader
from torchsummary import summary
from tqdm import tqdm
import matplotlib.pyplot as plt
from veg_dataset import FruitVegDataset

# 在此文当中定义一个BasicBlock类，继承nn.Module，适用于resnet18和resnet34的残差结构
class BasicBlock(nn.Module):
    # 指定扩张因子为1，主分支的卷积核个数不发生变化
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, is_downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = is_downsample
        if stride != 1 or in_channels != out_channels:  # 如果步幅不为1或者输入通道数和输出通道数不匹配，则需要下采样
            self.downsample = nn.Sequential(  # 下采样操作 1x1卷积核，步幅为stride，不使用偏置
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x):
        identity = x    # 保存输入数据，用于残差连接
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)   # 下采样操作
        out += identity
        out = self.relu(out)
        return out

# 在此文当中定义一个Bottleneck类，继承nn.Module，适用于resnet50,resnet101,resnet152的残差结构
class Bottleneck(nn.Module):
    # 指定扩张因子为4，主分支的卷积核个数发生变化
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, is_downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        self.relu = nn.ReLU()
        self.downsample = is_downsample
        if stride != 1 or in_channels != out_channels*self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels*self.expansion)
            )
    def forward(self, x):
        identity = x    # 保存输入数据，用于残差连接
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)   # 下采样操作
        out += identity
        out = self.relu(out)
        return out

# 定义ResNet类 用于实现resnet网络架构部分
class ResNet(nn.Module):
    def __init__(self, block, blocks_num, num_classes=1000, include_top=True):
        '''
        block: 残差块类型 BasicBlock 或 Bottleneck
        blocks_num: 每个残差块的层数列表 如resent18为[2, 2, 2, 2]，resent34为[3, 4, 6, 3]，resent50为[3, 4, 6, 3]，resent101为[3, 4, 23, 3]
        num_classes: 输出类别数
        include_top: 分类头，是否包含全连接层 如果为True，则包含全连接层，否则不包含
        '''
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 构建四个残差层，分别对应resnet的四个stage，每个stage的残差块个数为blocks_num[i]
        self.layer1 = self._make_layer(block, self.in_channels, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(self.in_channels, num_classes)  # 这里的in_channels已经更新过了
        for m in self.modules():  # 遍历模型中的所有模块
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, block_num, stride=1):
        '''
        block: 残差块类型 BasicBlock 或 Bottleneck
        out_channels: 残差结构中第一个卷积层的个数
        block_num: 每个残差块的层数
        stride: 步幅
        '''
        downsample = None
        # 如果步幅不为1 或者 输入通道与数残差输入通道数不匹配，则需要下采样
        if stride != 1 or self.in_channels != out_channels*block.expansion:
            # 对resnet18和resnet34来说，不满足条件，跳过下采样
            # 对resnet50,resnet101,resnet152来说，满足条件，需要下采样
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels*block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels*block.expansion)    #需要对齐便于进行残差连接，用expansion因子
            )
        layers = [] # 用于存储每个残差块的输出结果
        # 构建每个残差块
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels*block.expansion   # 更新输入通道数 对18和34来说in_channels保持不变，对50,101,152来说in_channels需要乘以expansion因子
        # 构建剩余的残差块
        for _ in range(1, block_num):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)   # 返回一个包含所有残差块的Sequential容器

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
        return x

    def resnet18(self, num_classes=1000, include_top=True, pretrained=False):
        '''
        num_classes: 输出类别数
        include_top: 分类头，是否包含全连接层 如果为True，则包含全连接层，否则不包含
        pretrained: 是否使用预训练权重 如果为True，则使用预训练权重，否则不使用
        '''
        return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, include_top=include_top)

    def resnet34(self, num_classes=1000, include_top=True, pretrained=False):
        return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)

    def resnet50(self, num_classes=1000, include_top=True, pretrained=False):
        return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)

    def resnet101(self, num_classes=1000, include_top=True, pretrained=False):
        return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)
        

if __name__ == "__main__":
    # # 测试BasicBlock网络
    # block = BasicBlock(in_channels=64, out_channels=64, stride=1)
    # print(block)
    # x = torch.randn(1, 64, 56, 56)
    # y = block(x)
    # print(y.shape)
    # print("--------------------------------")

    # # 测试Bottleneck网络
    # block = Bottleneck(in_channels=64, out_channels=64, stride=1)
    # print(block)
    # x = torch.randn(1, 64, 56, 56)
    # y = block(x)
    # print(y.shape)
    # print("--------------------------------")

    # # 测试resent34网络
    # net = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=1000, include_top=True)
    # print(net)
    # x = torch.randn(1, 3, 224, 224)
    # y = net(x)
    # print(y.shape)

    # 参数设置
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    batch_size = 256
    num_classes = 25
    epochs = 80
    init_lr = 1e-3
    current_lr = init_lr
    isLoad = False
    isWeight = True

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    best_val_accuracy = 0.0
    best_val_loss = float('inf')

    # 修改数据集路径
    train_path = Path("./Vegetable_Detection/train") # 为了健壮性这么写 不然macOS直接写路径就行
    test_path = Path("./Vegetable_Detection/test")
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
        # 使用更常用的ImageNet标准化参数
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 准备数据集 替换成自定义的FruitVegDataset 类 而不用ImageFolder
    train_dataset = FruitVegDataset(
        root_dir=train_path,
        transform=my_transform
    )

    val_dataset = FruitVegDataset(
        root_dir=test_path,
        transform=my_transform
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,           # 新增：开启 16 个子进程加载数据
        pin_memory=True          # 新增：加速数据转移到显卡
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=16,           # 新增
        pin_memory=True          # 新增
    )

    # 开始训练
    model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=True).to(device)
    # print(model)
    # 根据不同类别的训练数据个数，给一个权重系数（即”类别平衡“）
    # 多的数据 系数小一点，少的数据 系数大一点 
    if isWeight:
        # 基于精准样本量计算，并对表现差的类别（10, 24, 16, 21等）进行了 2-3 倍的额外增益
        class_weights = torch.tensor(
            [0.5769, 0.5052, 0.3713, 0.2684, 0.3329, 0.3518, 3.3347, 3.0985, 4.8288, 4.0636, 
             16.4038, # Class 10 (原20%Acc) -> 极大增强
             4.6770, 2.5467, 2.3385, 
             5.3974,  # Class 14
             2.3312, 
             5.1286,  # Class 16 (原44%Acc) -> 增强
             1.7375, 1.1319, 2.4954, 
             4.0810,  # Class 20
             13.9433, # Class 21 (样本最少) -> 极大增强
             9.2314,  # Class 22
             0.3733, 
             9.2035   # Class 24 (原30%Acc) -> 极大增强
            ], dtype=torch.float32
        )
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    # 设置优化器
    # 设置优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)

    # --- 修改：换成余弦退火调度器 ---
    # T_max 建议设为总 Epoch 数（比如 60 或 80）
    # eta_min 是学习率降到的最低值，设为 1e-6 确保它不会变成 0
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

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
            label_idx = torch.argmax(label, dim=1)

            pred_y = model(input_x)
            loss = criterion(pred_y, label_idx)
            loss.backward()
            optimizer.step()

            pred = torch.argmax(pred_y, dim=1)
            accuracy = torch.sum(pred == label_idx)/label.size(0)

            # 整体指标
            train_loss_epoch += loss.item() * label.size(0)
            train_acc_epoch += accuracy.item() * label.size(0)
            total_train_samples += label.size(0)

            # 统计每个类别的准确率
            for i in range(num_classes):
                train_class_correct[i] += ((pred == i) & (label_idx == i)).sum().item()
                train_class_total[i] += (label_idx == i).sum().item()

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
                label_idx = torch.argmax(label, dim=1)

                pred_y = model(input_x)
                loss = criterion(pred_y, label_idx)

                pred = torch.argmax(pred_y, dim=1)
                accuracy = torch.sum(pred == label_idx)/label.size(0)

                # 整体指标
                val_loss_epoch += loss.item() * label.size(0)
                val_acc_epoch += accuracy.item() * label.size(0)
                total_val_samples += label.size(0)

                # 统计每个类别的准确率
                for i in range(num_classes):
                    val_class_correct[i] += ((pred == i) & (label_idx == i)).sum().item()
                    val_class_total[i] += (label_idx == i).sum().item()

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
            print(f"[*]Best validation accuracy updated: {best_val_accuracy:.2%}, best validation loss: {best_val_loss:.4f}")
        
        # --- 新增：更新学习率 ---
        scheduler.step()
        # 顺便打印一下当前的学习率，方便你监控
        print(f"Epoch {epoch+1} finished. Current Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
    
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
    plt.savefig('resnet34_result.png')
    plt.show()
    plt.close()
    

    # 可视化显示几个分类正确的图像和对应的标签
    # ---------------------------------------------------------
    # 新增功能：随机可视化 24 张图像的真实标签与预测标签
    # ---------------------------------------------------------
    print("开始随机提取验证集图像进行可视化...")
    
    # 加载表现最好的模型权重
    if os.path.exists('best.pt'):
        checkpoint = torch.load('best.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"成功加载最佳权重，当时验证集准确率: {checkpoint['best_val_accuracy']:.2%}")
    
    model.eval()

    # 1. 设置展示 24 张图片
    num_to_show = 24
    # 重新定义一个 batch 为 24 的 loader 用于展示
    vis_loader = DataLoader(dataset=val_dataset, batch_size=num_to_show, shuffle=True)
    
    # 获取随机批次
    images, labels = next(iter(vis_loader))
    images = images.to(device)
    true_labels = torch.argmax(labels.to(device), dim=1)

    # 2. 模型预测
    with torch.no_grad():
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

    # 3. 反标准化参数
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)

    # 4. 绘图：设置 4 行 6 列
    plt.figure(figsize=(20, 15)) # 调大画布尺寸，保证 24 张图清晰
    for i in range(num_to_show):
        img = images[i] * std + mean
        img = torch.clamp(img, 0, 1)
        img = img.cpu().permute(1, 2, 0).numpy()

        t_label = true_labels[i].item()
        p_label = preds[i].item()

        ax = plt.subplot(4, 6, i + 1) # 4x6 布局
        ax.imshow(img)
        
        # 正确绿色，错误红色
        color = 'green' if t_label == p_label else 'red'
        ax.set_title(f"T:{t_label} | P:{p_label}", color=color, fontsize=12, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('random_predictions_24.png')
    print("可视化完成！请查看 random_predictions_24.png")
    plt.show()
    plt.close()