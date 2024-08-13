import torch
import torchvision 
from tqdm import tqdm


class Net2(torch.nn.Module):
    def __init__(self):
        super(Net2,self).__init__()
        self.backbone = torch.nn.Sequential(
            # 输入层：28x28x1
            torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            # 第二个卷积层：14x14x16
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            # 第三个卷积层：7x7x32
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            
            # Flatten层
            torch.nn.Flatten(),
            
            # 全连接层
            torch.nn.Linear(in_features=7 * 7 * 64, out_features=128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            
            torch.nn.Linear(in_features=128, out_features=10)
        )
        
    def forward(self,input):
        output = self.backbone(input)
        output = output.softmax(dim=1)
        return output
