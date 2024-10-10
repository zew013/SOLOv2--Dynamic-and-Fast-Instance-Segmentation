from torch import nn
import torch

class CNN(nn.Module):
    def __init__(self, in_channels, kernel_size = 3, padding = 1, channels = 256, stride=1):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=32, num_channels=channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.gn2 = nn.GroupNorm(num_groups=32, num_channels=channels)

        self.conv3 = nn.Conv2d(channels, channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.gn3 = nn.GroupNorm(num_groups=32, num_channels=channels)

        self.conv4 = nn.Conv2d(channels, channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.gn4 = nn.GroupNorm(num_groups=32, num_channels=channels)

        self.conv5 = nn.Conv2d(channels, channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.gn5 = nn.GroupNorm(num_groups=32, num_channels=channels)

        self.conv6 = nn.Conv2d(channels, channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.gn6 = nn.GroupNorm(num_groups=32, num_channels=channels)

        self.conv7 = nn.Conv2d(channels, channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.gn7 = nn.GroupNorm(num_groups=32, num_channels=channels)
    
    def forward(self, x):
        x = nn.ReLU()(self.gn1(self.conv1(x)))
        x = nn.ReLU()(self.gn2(self.conv2(x)))
        x = nn.ReLU()(self.gn3(self.conv3(x)))
        x = nn.ReLU()(self.gn4(self.conv4(x)))
        x = nn.ReLU()(self.gn5(self.conv5(x)))
        x = nn.ReLU()(self.gn6(self.conv6(x)))
        x = nn.ReLU()(self.gn7(self.conv7(x)))
        return x

class Category(nn.Module):
    def __init__(self, C=4, num_levels=5, channels = 256, kernel_size = 3, padding = 1):
        super(Category, self).__init__()
        self.conv= CNN(channels)
        self.output = nn.ModuleList([nn.Conv2d(channels, C-1, kernel_size, stride=1, padding=padding, bias=True) for _ in range(num_levels)])
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, level):
        x = self.conv(x)  
        x = self.sigmoid(self.output[level](x))
        return x 
       
    
class Mask(nn.Module):
    def __init__(self, num_grids, channels=256, kernel_size = 3):
        super(Mask, self).__init__()

        self.conv = CNN(channels+2)
        self.output = nn.ModuleList([nn.Conv2d(channels, num_grid*num_grid, kernel_size=1, stride=1, padding=0, bias=True) for num_grid in num_grids])
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, feature, x, y, level):
        batch_size = feature.shape[0]
        h, w = feature.shape[-2:] 
        x_expanded = x.expand(batch_size, 1, h, w)  
        y_expanded = y.expand(batch_size, 1, h, w)
        input_feature = torch.cat([feature, x_expanded, y_expanded], dim=1)
        x = self.conv(input_feature)
        x = self.sigmoid(self.output[level](x))
        return x
