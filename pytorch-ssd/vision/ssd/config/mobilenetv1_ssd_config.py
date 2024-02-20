import numpy as np

from vision.utils.box_utils import SSDSpec, SSDBoxSizes, generate_ssd_priors

# SSD 모델 구축 초기 설정
# 해당 모델의 사전 계산된 박스 생성 (이미지 내 객체 식별)

# 1. SSD 모델 초기 설정
image_size = 300
image_mean = np.array([127, 127, 127])  # RGB layout 평균
image_std = 128.0 # 표준편차
iou_threshold = 0.45 # IoU 임계값 (박스 간 겹침을 평가)
# 박스 중심과 크기의 변동성
center_variance = 0.1
size_variance = 0.2

# 2. SSD 스펙 설정
specs = [
    SSDSpec(19, 16, SSDBoxSizes(60, 105), [2, 3]),
    SSDSpec(10, 32, SSDBoxSizes(105, 150), [2, 3]),
    SSDSpec(5, 64, SSDBoxSizes(150, 195), [2, 3]),
    SSDSpec(3, 100, SSDBoxSizes(195, 240), [2, 3]),
    SSDSpec(2, 150, SSDBoxSizes(240, 285), [2, 3]),
    SSDSpec(1, 300, SSDBoxSizes(285, 330), [2, 3])
]

# 사전 계산 배열 (초기 박스)
# 3. SSD 박스 생성
priors = generate_ssd_priors(specs, image_size)

#print(' ')
#print('SSD-Mobilenet-v1 priors:')
#print(priors.shape)
#print(priors)
#print(' ')

#import torch
#torch.save(priors, 'mb1-ssd-priors.pt')

#np.savetxt('mb1-ssd-priors.txt', priors.numpy())