import torch.nn as nn

# VGG 네트워크 구조 생성 함수
def vgg(cfg, batch_norm=False):
    layers = []
    # 입력 채널 수 3 (RGB)
    in_channels = 3
    for v in cfg: # 리스트 순회하면서
        if v == 'M': # 'M'이면 Max Pooling 레이어를 추가
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)] # 2x2 커널 크기, stride 2
        elif v == 'C': # 정수가 되도록
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        # 둘 다 아니라면, 3x3 커널을 사용하는 Convolutional 레이어를 추가
        else: # v로의 컨볼루션을 수행
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers