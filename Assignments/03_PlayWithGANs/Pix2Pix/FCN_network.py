import torch
import torch.nn as nn
import math


class Generator(nn.Module):
    def __init__(self, noise_dim=100, condition_dim=3, output_dim=3, dropout_rate=0.5):
        """
        Args:
            noise_dim: 噪声向量 z 的维度
            condition_dim: 条件输入 x 的通道数（例如 RGB 图像为 3 通道）
            output_dim: 生成的伪造数据 y 的通道数（例如 RGB 图像为 3 通道）
        """
        super(Generator, self).__init__()
        # 噪声和条件拼接后的输入通道数
        input_dim = noise_dim + condition_dim

        # 网络结构：卷积-反卷积
        self.net = nn.Sequential(
            nn.Conv2d(input_dim, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),  # 添加 Dropout 层以减少过拟合

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),  # 添加 Dropout 层以减少过拟合

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),  # 添加 Dropout 层以减少过拟合

            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),  # 添加 Dropout 层以减少过拟合

            # nn.Conv2d(1024, 2048, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(2048),
            # nn.ReLU(inplace=True),
            # nn.Dropout(dropout_rate),  # 添加 Dropout 层以减少过拟合

            # nn.ConvTranspose2d(2048, 1024, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(1024),
            # nn.ReLU(inplace=True),
            # nn.Dropout(dropout_rate),  # 添加 Dropout 层以减少过拟合

            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),  # 添加 Dropout 层以减少过拟合

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),  # 添加 Dropout 层以减少过拟合

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),  # 添加 Dropout 层以减少过拟合

            nn.ConvTranspose2d(128, output_dim, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # 使用 Tanh 激活输出 [-1, 1] 的值
        )

    def forward(self, x, z):
        """
        Args:
            x: 条件输入 (N, condition_dim, H, W)
            z: 噪声向量 (N, noise_dim, 1, 1)

        Returns:
            output: 生成的伪造数据 (N, output_dim, H, W)
        """
        # 将条件输入和噪声拼接到通道维度
        z = z.expand(-1, -1, x.size(2), x.size(3))  # 将 z 的宽高扩展为与 x 匹配
        input = torch.cat([x, z], dim=1)  # 拼接后维度为 (N, condition_dim + noise_dim, H, W)
        return self.net(input)

    

class Discriminator(nn.Module):
    def __init__(self, condition_dim=3, target_dim=3, dropout_rate=0.5):
        """
        Args:
            condition_dim: 条件输入 x 的通道数（例如 RGB 图像为 3 通道）
            target_dim: 输入目标 y 的通道数（例如 RGB 图像为 3 通道）
        """
        super(Discriminator, self).__init__()
        # 条件和目标拼接后的输入通道数
        input_dim = condition_dim + target_dim

        # 网络结构：卷积
        self.net = nn.Sequential(
            nn.Conv2d(input_dim, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_rate),  # 添加 Dropout 层以减少过拟合

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_rate),  # 添加 Dropout 层以减少过拟合

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_rate),  # 添加 Dropout 层以减少过拟合

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_rate),  # 添加 Dropout 层以减少过拟合

            # nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(1024),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(dropout_rate),  # 添加 Dropout 层以减少过拟合

            # nn.Conv2d(1024, 2048, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(2048),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(dropout_rate),  # 添加 Dropout 层以减少过拟合

            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0),  # 输出 1 个值表示真实概率
        )

        # # 全连接层，输入尺寸计算：假设输入图像尺寸为 64x64
        # feature_map_size = 16  # 经过多层卷积后特征图的大小
        # flattened_size = 512 * feature_map_size * feature_map_size  # 512 是最后一层的通道数
        # self.fc = nn.Linear(flattened_size, 1)  # 全连接层，输出一个值表示真实概率

    def forward(self, x, y):
        """
        Args:
            x: 条件输入 (N, condition_dim, H, W)
            y: 目标输入 (真实或伪造) (N, target_dim, H, W)

        Returns:
            output: 判别结果 (N, 1, 1, 1)
        """
        # 将条件输入和目标拼接到通道维度
        input = torch.cat([x, y], dim=1)  # 拼接后维度为 (N, condition_dim + target_dim, H, W)
        return torch.sigmoid(self.net(input))  # 输出为 [0, 1] 的概率
    
        # # 卷积网络
        # input = self.net(input)
        # # 扁平化为向量
        # input = input.view(input.size(0), -1)  # 扁平化
        # # 全连接层
        # input = self.fc(input)
        # return input
