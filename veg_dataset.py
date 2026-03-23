import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

class FruitVegDataset(Dataset):
    def __init__(self, root_dir="dataset/Vegetable/train", transform=None):
        """
        自定义数据集加载类
        :param root_dir: 训练集根目录
        :param transform: 外部传入的图像预处理和增强
        """
        self.root_dir = root_dir

        # 1. 读取标签CSV文件
        self.csv_path = os.path.join(root_dir, "_classes.csv")
        self.df = pd.read_csv(self.csv_path)

        # 2. 提取文件名（第一列）
        self.filenames = self.df.iloc[:, 0].values

        # 3. 提取标签（从第2列到最后一列 → 26个分类）
        self.labels = self.df.iloc[:, 1:].values

        # 用外部传入的transform
        self.transform = transform

    def __len__(self):
        # 返回数据集总样本数
        return len(self.filenames)

    def __getitem__(self, idx):
        # 1. 获取图片名称
        img_name = self.filenames[idx]
        img_path = os.path.join(self.root_dir, img_name)

        # 2. 读取图片（PIL 格式）
        image = Image.open(img_path).convert("RGB")

        # 3. 获取对应标签（转为 tensor）
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        # 4. 图像预处理（resize + 张量转换等）
        if self.transform:
            image = self.transform(image)

        return image, label

# ------------------------------------------------------------------------------
# 【使用示例】定义预处理 + 创建数据集
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # 图像预处理：416*416 → resize 到 224*224 + 转tensor

    # 创建数据集
    train_dataset = FruitVegDataset(
        root_dir="./Vegetable Detecttion/train"
    )

    # # 测试：取一个样本查看形状
    # img, label = train_dataset[0]
    # print("图像形状:", img.shape)    # torch.Size([3, 224, 224])
    # print("标签长度:", label.shape)  # torch.Size([26]) → 26分类
    # print("数据集总数量:", len(train_dataset))

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=8,
        shuffle=True,  # 训练建议打乱
        num_workers=0  # Windows 下如果报错就设为0
    )

    # 3. 遍历 DataLoader，打印每个批次的形状
    print("开始遍历每个批次...\n")
    for (images, labels) in train_loader:
        print(f"  图像特征形状: {images.shape}")  # [batch, 3, 224, 224]
        print(f"  标签形状:     {labels.shape}\n")  # [batch, 26]

