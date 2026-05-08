# AI Learning Projects

PyTorch 与强化学习练习项目合集。

## 项目列表

### rl_learning — K 臂老虎机（强化学习）

`rl_learning/k_armed_bandit.py`

用 epsilon-greedy 策略解决 2 臂老虎机问题。两台机器奖励分别满足 N(500,50) 和 N(550,100)，
从乐观初始值 Q=998 出发，1000 步后收敛。

```bash
python rl_learning/k_armed_bandit.py
```

---

### cifar10_project — CIFAR-10 图像分类（CNN）

`cifar10_project/`

基于 LeNet 框架搭建简单 CNN，在 CIFAR-10（10 类，50000 张训练图）上训练，
使用 Adam 优化器，10 轮训练。数据集首次运行时自动下载至 `cifar10_project/data/`。

```bash
python cifar10_project/main.py   # 训练
python cifar10_project/show.py   # 可视化预测结果
```

## 环境

```bash
pip install torch torchvision torchsummary matplotlib numpy
```

- 设备：M4 Mac（MPS）/ CUDA / CPU 自动选择
- 框架：PyTorch
