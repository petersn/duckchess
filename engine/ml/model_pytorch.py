import torch

class ConvBlock(torch.nn.Module):
    def __init__(self, filters, kernel_size=3):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(filters, filters, kernel_size=kernel_size, padding="same")
        self.conv2 = torch.nn.Conv2d(filters, filters, kernel_size=kernel_size, padding="same")
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        skip0 = x
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = x + skip0
        return x

class Flatten(torch.nn.Module):
    def forward(self, x):
        # x has shape [batch, channel, height, width]
        n, _, _, _ = x.shape
        return x.reshape((n, -1))

class Network(torch.nn.Module):
    def __init__(
        self,
        blocks=12,
        feature_count=128,
        final_features=32,
        input_channels=22,
        policy_outputs=64 * 64,
        value_outputs=1,
    ):
        super().__init__()
        layers = [
            torch.nn.Conv2d(input_channels, feature_count, kernel_size=3, padding="same"),
        ] + [
            ConvBlock(feature_count) for _ in range(blocks)
        ] + [
            torch.nn.Conv2d(feature_count, final_features, kernel_size=3, padding="same"),
            torch.nn.ReLU(),
            Flatten(),
        ]
        self.conv_tower = torch.nn.Sequential(*layers)
        self.policy_dense = torch.nn.Linear(64 * final_features, policy_outputs)
        self.value_dense = torch.nn.Linear(64 * final_features, value_outputs)

    def forward(self, x):
        deep_embedding = self.conv_tower(x)
        policy = self.policy_dense(deep_embedding)
        value = torch.tanh(self.value_dense(deep_embedding))
        return policy, value
