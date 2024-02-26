import torch
from torch.nn import Conv2d, Sequential, ModuleList, ReLU, BatchNorm2d
from ..nn.mobilenet import MobileNetV1

from .ssd import SSD
from .predictor import Predictor
from .config import mobilenetv1_ssd_config as config

# 기존의 convolution을 두 단계로 seperate해서 수행하는 것: 파라미터 수 줄이기
# 먼저 depthwise convolution (깊이별 컨볼루션)이 이루어진 후 pointwise convolution (포인트별 컨볼루션) 수행
def SeperableConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0):
    return Sequential(
        # depthwise convolution
        Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
               groups=in_channels, stride=stride, padding=padding),
        ReLU(),
        # pointwise convolution
        Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
    )


def create_mobilenetv1_ssd_lite(num_classes, is_test=False):
    # 1001개의 클래스로 사전 훈련된 모델을 생성하고, 그중 dropout layer 비활성한 모델을 base_net 설정
    base_net = MobileNetV1(1001).model  # disable dropout layer
    # SSD 모델의 특징을 가져올 base 네트워크의 layer index 정의
    source_layer_indexes = [
        12,
        14,
    ]
    extras = ModuleList([
        Sequential(
            Conv2d(in_channels=1024, out_channels=256, kernel_size=1),
            ReLU(),
            SeperableConv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
        ),
        Sequential(
            Conv2d(in_channels=512, out_channels=128, kernel_size=1),
            ReLU(),
            SeperableConv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
        ),
        Sequential(
            Conv2d(in_channels=256, out_channels=128, kernel_size=1),
            ReLU(),
            SeperableConv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
        ),
        Sequential(
            Conv2d(in_channels=256, out_channels=128, kernel_size=1),
            ReLU(),
            SeperableConv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        )
    ])
    # 각 피쳐 레이어에 대한 회귀와 분류 헤더 정의
    regression_headers = ModuleList([
        SeperableConv2d(in_channels=512, out_channels=6 * 4, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=1024, out_channels=6 * 4, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=512, out_channels=6 * 4, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=6 * 4, kernel_size=1),
    ])

    classification_headers = ModuleList([
        SeperableConv2d(in_channels=512, out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=1024, out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=512, out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=1),
    ])

    return SSD(num_classes, base_net, source_layer_indexes,
               extras, classification_headers, regression_headers, is_test=is_test, config=config)


def create_mobilenetv1_ssd_lite_predictor(net, candidate_size=200, nms_method=None, sigma=0.5, device=None):
    predictor = Predictor(net, config.image_size, config.image_mean,
                          config.image_std,
                          nms_method=nms_method,
                          iou_threshold=config.iou_threshold,
                          candidate_size=candidate_size,
                          sigma=sigma,
                          device=device)
    return predictor