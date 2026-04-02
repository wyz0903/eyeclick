"""
视线追踪 - 卷积残差神经网络训练脚本 (Gaze Mapping ResNet)

该脚本读取 dataset_collector.py 收集的 CSV 数据，
使用 1D-CNN 残差网络，仅基于 478 个原始 3D 面部特征点来预测屏幕绝对坐标。

依赖项:
    - torch, pandas, numpy, scikit-learn
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import copy
import os

# ================= 统一超参数配置区域 =================
CONFIG = {
    'dataset_path': 'gaze_dataset.csv',  # 数据集路径
    'model_save_path': 'gaze_resnet_model.pth',  # 最佳模型保存路径
    'batch_size': 16,  # 批次大小
    'learning_rate': 0.0005,  # 学习率
    'epochs': 500,  # 最大训练轮数
    'patience': 30,  # 早停机制的容忍轮数 (Patience)
    'val_split': 0.2,  # 验证集比例
    'random_seed': 42  # 随机种子
}
# ======================================================

# 固定随机种子以保证结果可复现
torch.manual_seed(CONFIG['random_seed'])
np.random.seed(CONFIG['random_seed'])


class ResidualBlock(nn.Module):
    """
    自定义残差块。
    包含：卷积层 -> 批归一化层 -> ReLU激活 -> 池化层。
    由于池化层会改变特征图的尺寸，Shortcut 捷径连接也包含了对应的变换。
    """

    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        # 1. 卷积层 (保持空间长度不变)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        # 2. 批归一化层
        self.bn = nn.BatchNorm1d(out_channels)
        # 3. ReLU 激活函数
        self.relu = nn.ReLU()
        # 4. 池化层 (特征图长度减半)
        self.pool = nn.MaxPool1d(kernel_size=2)

        # 捷径连接 (Shortcut)：调整通道数和空间长度以对齐相加维度
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.MaxPool1d(kernel_size=2)
        )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.pool(out)
        # 残差相加
        out += identity
        return out


class GazeResNet(nn.Module):
    """视线映射残差卷积神经网络"""

    def __init__(self):
        super(GazeResNet, self).__init__()
        # 输入形状: (Batch, 3, 478)

        # 包含 3 个残差块
        self.layer1 = ResidualBlock(in_channels=3, out_channels=64)  # 输出长度: 478 / 2 = 239
        self.layer2 = ResidualBlock(in_channels=64, out_channels=128)  # 输出长度: 239 / 2 = 119
        self.layer3 = ResidualBlock(in_channels=128, out_channels=256)  # 输出长度: 119 / 2 = 59

        # 展平层
        self.flatten = nn.Flatten()

        # 全连接层输出回归结果
        self.fc1 = nn.Linear(256 * 59, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 2)  # 输出预测的 target_x, target_y

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class GazeDataset(Dataset):
    """自定义 PyTorch 数据集"""

    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class EarlyStopping:
    """早停机制：当验证集 Loss 在连续 patience 轮中没有下降时，提前停止训练"""

    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_wts = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_wts = copy.deepcopy(model.state_dict())
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model_wts = copy.deepcopy(model.state_dict())
            self.counter = 0


def prepare_data(csv_path):
    """
    读取并预处理数据。
    剥离高级特征，仅保留原始特征点，并重塑张量维度。
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到数据集文件: {csv_path}。请先运行采集脚本。")

    df = pd.read_csv(csv_path)

    # 获取特征列名称 (仅保留 lm_0_x 到 lm_477_z 的原始特征)
    raw_feature_cols = [col for col in df.columns if col.startswith('lm_')]

    # 提取特征 X 和标签 Y
    X_raw = df[raw_feature_cols].values
    y = df[['target_x', 'target_y']].values

    # 重塑 X 的维度以适配 Conv1d: (样本数, 通道数, 序列长度)
    # 1. 先重塑为 (N, 478个点, 3个坐标轴)
    # 2. 维度转置为 (N, 3个坐标轴, 478个点)
    X = X_raw.reshape(-1, 478, 3).transpose(0, 2, 1)

    # 对标签 Y (屏幕坐标) 进行标准化，有助于加速回归模型的收敛
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y)

    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_scaled, test_size=CONFIG['val_split'], random_state=CONFIG['random_seed']
    )

    return X_train, X_val, y_train, y_val, scaler_y


def main():
    print("--- 开始准备数据集 ---")
    X_train, X_val, y_train, y_val, scaler_y = prepare_data(CONFIG['dataset_path'])

    train_dataset = GazeDataset(X_train, y_train)
    val_dataset = GazeDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

    print(f"训练集大小: {len(train_dataset)} | 验证集大小: {len(val_dataset)}")
    print(f"输入特征维度: {X_train.shape}")

    # 检查硬件设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用的计算设备: {device}")

    # 初始化模型、损失函数和优化器
    model = GazeResNet().to(device)
    criterion = nn.MSELoss()  # 回归任务使用均方误差
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])

    # 初始化早停机制
    early_stopping = EarlyStopping(patience=CONFIG['patience'])

    print("\n--- 开始训练模型 ---")
    for epoch in range(CONFIG['epochs']):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)

        train_loss /= len(train_loader.dataset)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)

        val_loss /= len(val_loader.dataset)

        print(f"Epoch [{epoch + 1:03d}/{CONFIG['epochs']}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # 调用早停机制进行监测
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f"\n触发早停机制！连续 {CONFIG['patience']} 轮验证集损失未下降，训练提前结束。")
            break

    # 加载最佳模型权重并保存
    print("\n--- 训练结束 ---")
    if early_stopping.best_model_wts:
        model.load_state_dict(early_stopping.best_model_wts)

    torch.save(model.state_dict(), CONFIG['model_save_path'])
    print(f"最佳模型已保存至: {CONFIG['model_save_path']}")
    print(f"最佳验证集损失 (MSE): {early_stopping.best_loss:.4f}")


if __name__ == "__main__":
    main()