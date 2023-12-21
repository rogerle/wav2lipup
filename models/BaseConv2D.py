from torch import nn


class BaseConv2D(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, act='relu', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(cout)
        )
        if act == 'relu':
            self.act = nn.ReLU()
        if act == 'prelu':
            self.act = nn.PReLU()
        if act == 'leaky':
            self.act = nn.LeakyReLU(0.01, inplace=True)
        if act == 'sigmoid':
            self.act = nn.Sigmoid()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)
