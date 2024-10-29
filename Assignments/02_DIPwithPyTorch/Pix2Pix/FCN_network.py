import torch.nn as nn


class FullyConvNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder (Convolutional Layers)
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                3, 8, kernel_size=4, stride=2, padding=1
            ),  # Input channels: 3, Output channels: 8
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                8, 32, kernel_size=4, stride=2, padding=1
            ),  # Input channels: 8, Output channels: 16
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                32, 128, kernel_size=4, stride=2, padding=1
            ),  # Input channels: 16, Output channels: 32
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )


        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(
                128, 32, kernel_size=4, stride=2, padding=1
            ),  # Input channels: 32, Output channels: 16
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.deconv6 = nn.Sequential(
            nn.ConvTranspose2d(
                32, 8, kernel_size=4, stride=2, padding=1
            ),  # Input channels: 16, Output channels: 8
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        )
        self.deconv7 = nn.Sequential(
            nn.ConvTranspose2d(
                8, 3, kernel_size=4, stride=2, padding=1
            ),  # Input channels: 8, Output channels: 3
            nn.Tanh(),  # Use Tanh activation function for RGB output
        )

        ### FILL: add ConvTranspose Layers
        ### None: since last layer outputs RGB channels, may need specific activation function

    def forward(self, x):
        # Encoder forward pass
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # x = self.conv4(x)
        # x = self.conv5(x)
        # x = self.conv6(x)
        # x = self.conv7(x)

        # x = self.deconv1(x)
        # x = self.deconv2(x)
        # x = self.deconv3(x)
        # x = self.deconv4(x)
        x = self.deconv5(x)
        x = self.deconv6(x)
        output = self.deconv7(x)

        # Decoder forward pass

        ### FILL: encoder-decoder forward pass

        return output
