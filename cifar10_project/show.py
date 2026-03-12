import torch
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict

# -------------------- 配置 --------------------
pth_path = "best_model.pth"          # 替换为你的 .pth 文件路径
plot_histograms = True                # 是否绘制直方图
plot_layer_images = True               # 是否绘制特定层的权重图像（如卷积核）
layers_to_plot = ['fc.weight', 'conv1.weight']  # 指定要绘制图像的层名（根据实际名称修改）
# ---------------------------------------------

def load_state_dict(pth_path):
    """加载 .pth 文件，返回 state_dict (OrderedDict)"""
    checkpoint = torch.load(pth_path, map_location='cpu')
    if isinstance(checkpoint, dict):
        # 常见情况：可能是完整的 checkpoint，包含 'state_dict' 或 'model' 键
        if 'state_dict' in checkpoint:
            return checkpoint['state_dict']
        elif 'model' in checkpoint:
            return checkpoint['model']
        else:
            # 假设直接就是 state_dict
            return checkpoint
    else:
        raise TypeError("无法识别的 .pth 文件格式，请确保它包含 state_dict")

state_dict = load_state_dict(pth_path)

# 1. 打印统计信息
print("参数统计信息：")
for name, param in state_dict.items():
    if param.ndim >= 1:  # 只考虑有数据的参数（避免标量）
        stats = {
            'mean': param.mean().item(),
            'std': param.std().item(),
            'max': param.max().item(),
            'min': param.min().item(),
        }
        print(f"{name:40s} shape: {str(param.shape):20s} "
              f"mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
              f"max={stats['max']:.4f}, min={stats['min']:.4f}")

# 2. 绘制直方图（所有层）
if plot_histograms:
    num_layers = len([n for n, p in state_dict.items() if p.ndim >= 1])
    cols = 4
    rows = (num_layers + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*3))
    axes = axes.flatten() if num_layers > 1 else [axes]
    fig.suptitle('Weight Histograms of All Layers', fontsize=16)

    for idx, (name, param) in enumerate(state_dict.items()):
        if param.ndim < 1:
            continue
        ax = axes[idx]
        ax.hist(param.flatten().detach().cpu().numpy(), bins=50, alpha=0.7, color='steelblue')
        ax.set_title(f"{name[:20]}...")
        ax.set_xlabel('Weight value')
        ax.set_ylabel('Frequency')
        ax.grid(True, linestyle='--', alpha=0.5)

    # 隐藏多余的子图
    for j in range(idx+1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

# 3. 绘制特定层的权重图像（热力图/卷积核）
if plot_layer_images and layers_to_plot:
    for layer_name in layers_to_plot:
        if layer_name not in state_dict:
            print(f"警告：层 '{layer_name}' 不存在于 state_dict 中")
            continue

        param = state_dict[layer_name]
        if param.ndim == 2:  # 全连接层权重 (out_features, in_features)
            # 绘制热力图
            plt.figure(figsize=(8, 6))
            plt.imshow(param.detach().cpu().numpy(), aspect='auto', cmap='viridis')
            plt.colorbar(label='Weight value')
            plt.title(f'Weight matrix: {layer_name} ({param.shape})')
            plt.xlabel('Input features')
            plt.ylabel('Output features')
            plt.show()

        elif param.ndim == 4:  # 卷积层权重 (out_channels, in_channels, kH, KW)
            # 选取前 16 个卷积核绘制为小图像（如果 out_channels > 16 则取前 16 个）
            weights = param.detach().cpu().numpy()
            out_c = weights.shape[0]
            plot_c = min(out_c, 16)
            cols = 4
            rows = (plot_c + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
            axes = axes.flatten() if plot_c > 1 else [axes]
            fig.suptitle(f'Convolution kernels: {layer_name} (first {plot_c} out_channels)', fontsize=14)

            for i in range(plot_c):
                ax = axes[i]
                # 如果是多输入通道，取所有通道的平均值展示单张图，或者只展示第一个通道
                if weights.shape[1] > 1:
                    # 显示每个 out_channel 的 in_channel 平均
                    kernel_img = weights[i].mean(axis=0)
                else:
                    kernel_img = weights[i, 0]
                ax.imshow(kernel_img, cmap='gray')
                ax.set_title(f'oc {i}')
                ax.axis('off')

            for j in range(plot_c, len(axes)):
                axes[j].axis('off')

            plt.tight_layout()
            plt.show()

        else:
            print(f"层 '{layer_name}' 的维度为 {param.ndim}，暂不支持图像绘制（仅支持 2D 和 4D）")