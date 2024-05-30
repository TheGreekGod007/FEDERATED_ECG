import torch.nn as nn 

# Assuming you have defined Model, train, and evaluate functions
# Define your model architecture A and Model before loading the state dictionary

class A(nn.Module):
    def __init__(self):
        super().__init__()

        self.intro_bn = nn.BatchNorm1d(32)
        self.C11 = nn.Conv1d(32, 32, kernel_size=5, padding=2)
        self.A11 = nn.ReLU()
        self.C12 = nn.Conv1d(32, 32, kernel_size=5, padding=2)
        self.A12 = nn.ReLU()
        self.M11 = nn.MaxPool1d(kernel_size=5, stride=2)

    def forward(self, x):
        x = self.intro_bn(x)
        C = x
        x = self.C11(x)
        x = self.A11(x)
        x = self.C12(x)
        x = x + C
        x = self.A12(x)
        x = self.M11(x)
        return x

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_in = nn.Conv1d(1, 32, kernel_size=5)
        self.A_blocks = nn.ModuleList(A() for i in range(5))
        self.avg_pool = nn.AvgPool1d(2)
        self.fc1 = nn.Linear(32, 32)
        self.acc1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 5)



    def forward(self, x):
        x = self.conv_in(x)
        for i in range(5):
            x = self.A_blocks[i](x)
        x = self.avg_pool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.acc1(x)
        x = self.fc2(x)
        return x


