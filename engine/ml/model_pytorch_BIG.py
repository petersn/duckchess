import torch

class ConvBlock(torch.nn.Module):
    def __init__(self, filters, kernel_size=3):
        super().__init__()
        # Vageuly inspired by fixup initialization, see:
        # https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md
        # https://arxiv.org/pdf/1901.09321.pdf
        self.conv1 = torch.nn.Conv2d(filters, filters, kernel_size=kernel_size, padding="same")
        self.conv2 = torch.nn.Conv2d(filters, filters, kernel_size=kernel_size, padding="same")
        self.conv2.weight.data *= 1e-3
        self.activation = torch.nn.ReLU() #GELU()

    def forward(self, x):
        skip = x
        x = self.activation(x)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        return x + skip

class Flatten(torch.nn.Module):
    def forward(self, x):
        # x has shape [batch, channel, height, width]
        n, _, _, _ = x.shape
        return x.reshape((n, -1))

class Network(torch.nn.Module):
    def __init__(
        self,
        blocks=10,#20,
        feature_count=128,#384,
        input_channels=37,
        policy_channels=32,#64,
    ):
        super().__init__()
        layers = [
            torch.nn.Conv2d(input_channels, feature_count, kernel_size=3, padding="same"),
        ] + [
            ConvBlock(feature_count) for _ in range(blocks)
        ]
        self.conv_tower = torch.nn.Sequential(*layers)
        self.policy_head = torch.nn.Sequential(
            torch.nn.Conv2d(feature_count, policy_channels, kernel_size=3, padding="same"),
            Flatten(),
        )
        self.value_head = torch.nn.Sequential(
            torch.nn.Conv2d(feature_count, 8, kernel_size=3, padding="same"),
            Flatten(),
            torch.nn.ReLU(),
            torch.nn.Linear(8 * 8 * 8, 64),
            torch.nn.ReLU(),
            #torch.nn.Tanh(),
        )
        self.wdl_head = torch.nn.Linear(64, 3)
        self.mcts_value_prediction_head = torch.nn.Linear(64, 1)

    def forward(self, x):
        deep_embedding = self.conv_tower(x)
        policy = self.policy_head(deep_embedding)
        value_outputs = self.value_head(deep_embedding)
        wdl = self.wdl_head(value_outputs)
        mcts_value_prediction = self.mcts_value_prediction_head(value_outputs)
        return policy, wdl, torch.tanh(mcts_value_prediction)
