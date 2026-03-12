'''
常用工具函数
'''
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 如果以后切到 CUDA 环境，这里也兼容
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # elif torch.backends.mps.is_available():
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    # elif torch.backends.mps.is_available():
    #     return torch.device("mps")
    else:
        return torch.device("cpu")
    
def accuracy(output, target):
    '''计算一个batch的分类准确率'''
    pred = output.argmax(dim=1)
    correct = (pred == target).sum().item()
    acc = correct / target.size(0)
    return acc

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    '''训练'''
    model.train()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0
    
    for imgs, labels in dataloader:
        # 先送进训练处
        imgs = imgs.to(device)
        labels = labels.to(device)
        # 前向传播
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 统计
        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        running_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size
    epoch_loss = running_loss / total_samples
    epoch_acc = running_correct / total_samples

    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device):
    '''测试'''
    with torch.no_grad():
        model.eval()
        running_loss = 0.0
        running_correct = 0
        total_samples = 0
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)
            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            running_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total_samples += batch_size
        epoch_loss = running_loss / total_samples
        epoch_acc = running_correct / total_samples
        return epoch_loss, epoch_acc
    
def save_model(model, save_path="best_model.pth"):
    torch.save(model.state_dict(), save_path)
    print(f"模型保存到：{save_path}")

def plot_curves(train_losses, val_losses, train_accs, val_accs):
    '''绘制loss和acc曲线'''
    plt.figure(figsize=(12,5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='train_loss')
    plt.plot(val_losses,label='val_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss Curve')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='train_acc')
    plt.plot(val_accs,label='val_acc')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy Curve')
    plt.legend()

    plt.tight_layout()
    plt.show()



















