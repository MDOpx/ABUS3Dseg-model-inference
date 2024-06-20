import torch
import torch.nn as nn
import pdb
# 입력 feature의 크기: [batch_size, in_channels, feature_size]
input_feature = torch.randn(2, 1, 320)  # 채널 크기는 1이고 feature 크기는 320입니다.

# 1D convolution을 사용하여 특징 크기를 줄이는 모델 정의
class ConvolutionModel(nn.Module):
    def __init__(self):
        super(ConvolutionModel, self).__init__()
        # 1D convolution 레이어 정의
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=9, stride=13, padding=1)# (2, 5, 25)
        self.conv1d2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=27, dilation=3, stride=2, padding=4)# (2, 5, 25)
        self.conv1d3 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=9, dilation=1, stride=3, padding=31)# (2, 5, 25)
        self.fc1 = nn.Linear(320, 25)

    def forward(self, x):
        # 입력 feature에 대해 1D convolution 적용
        x1 = self.conv1d(x)
        x2 = self.conv1d2(x)
        x3 = self.conv1d3(x)
        print(x1.shape)
        print(x2.shape)
        print(x3.shape)
        return x

# 모델 초기화
model = ConvolutionModel()

# 입력 feature에 모델 적용
output_feature = model(input_feature)

# 출력 feature의 크기 확인