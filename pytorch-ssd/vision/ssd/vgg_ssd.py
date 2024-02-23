import torch
from torch.nn import Conv2d, Sequential, ModuleList, ReLU, BatchNorm2d
from ..nn.vgg import vgg # VGG 네트워크

from .ssd import SSD
from .predictor import Predictor
from .config import vgg_ssd_config as config # SSD 모델 설정

# SSD 아키텍처 생성

# VGG-16 기반 SSD 모델 생성
def create_vgg_ssd(num_classes, is_test=False): # 탐지할 객체 클래스 수, 테스트 모드 사용 X
    # VGG 네트워크 구성 정의
    vgg_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
                  512, 512, 512] # VGG 네트워크의 레이어 구성
    base_net = ModuleList(vgg(vgg_config)) # 모듈 리스트로 만듦

    # SSD 모델에서 특정 레이어를 추출하는 데 사용할 인덱스 지정
    source_layer_indexes = [
        (23, BatchNorm2d(512)), # 23번째 레이어, Batch Normalization
        len(base_net), # 전체 레이어 개수 사용
    ]
    # 여러 scale에서 객체 검출하기 위해 extra 계층 정의 -> 하나의 모듈로
    extras = ModuleList([
        Sequential(
            Conv2d(in_channels=1024, out_channels=256, kernel_size=1),
            ReLU(),
            Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
            ReLU()
        ),
        Sequential(
            Conv2d(in_channels=512, out_channels=128, kernel_size=1),
            ReLU(),
            Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            ReLU()
        ),
        Sequential(
            Conv2d(in_channels=256, out_channels=128, kernel_size=1),
            ReLU(),
            Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            ReLU()
        ),
        Sequential(
            Conv2d(in_channels=256, out_channels=128, kernel_size=1),
            ReLU(),
            Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            ReLU()
        )
    ])

    regression_headers = ModuleList([
        Conv2d(in_channels=512, out_channels=4 * 4, kernel_size=3, padding=1),
        Conv2d(in_channels=1024, out_channels=6 * 4, kernel_size=3, padding=1),
        Conv2d(in_channels=512, out_channels=6 * 4, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=4 * 4, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=4 * 4, kernel_size=3, padding=1), # TODO: change to kernel_size=1, padding=0?
    ])

    classification_headers = ModuleList([
        Conv2d(in_channels=512, out_channels=4 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=1024, out_channels=6 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=512, out_channels=6 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=4 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=4 * num_classes, kernel_size=3, padding=1), # TODO: change to kernel_size=1, padding=0?
    ])

    return SSD(num_classes, base_net, source_layer_indexes,
               extras, classification_headers, regression_headers, is_test=is_test, config=config)

# SSD 모델에 대한 predictor 생성: 최종 객체 후보 상자 수 지정 및 파라미터 설정
def create_vgg_ssd_predictor(net, candidate_size=200, nms_method=None, sigma=0.5, device=None):
    predictor = Predictor(net, config.image_size, config.image_mean,
                          nms_method=nms_method,
                          iou_threshold=config.iou_threshold,
                          candidate_size=candidate_size,
                          sigma=sigma,
                          device=device)
    return predictor