import torch.nn as nn
import torch
import numpy as np
from typing import List, Tuple
import torch.nn.functional as F

from ..utils import box_utils
from collections import namedtuple
GraphPath = namedtuple("GraphPath", ['s0', 'name', 's1']) # 데이터 타입 정의

# SSD 클래스
class SSD(nn.Module):
    def __init__(
            self, num_classes: int, base_net: nn.ModuleList,
            source_layer_indexes: List[int], extras: nn.ModuleList,
            classification_headers: nn.ModuleList, regression_headers: nn.ModuleList,
            is_test=False, config=None, device=None
    ):
        super(SSD, self).__init__()

        # 클래스 수, 기본 (백본) 네트워크, 원본 레이어 인덱스,
        # 추가 레이어, 분류 헤더, 회귀 헤더, 테스트 여부 등 지정
        self.num_classes = num_classes
        self.base_net = base_net
        self.sourch_layer_indexes = source_layer_indexes
        self.extras = extras
        self.classification_headers = classification_headers
        self.regression_headers = regression_headers
        self.is_test = is_test
        self.config = config

        # 원본 레이어 인덱스 레이어를 모듈 리스트에 추가해 레이어들을 등록
        self.source_layer_add_ons = nn.ModuleList(
            [
                t[1]
                for t in source_layer_indexes
                if isinstance(t, tuple) and not isinstance(t, GraphPath)
            ]
        )
    
    # 순전파 함수
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        confidences = []
        locations = []
        start_layer_index = 0
        header_index = 0
        # 원본 레이어 인덱스의 레이어에서 위치 헤더와 분류 헤더 계산
        for end_layer_index in self.source_layer_indexes:
            if isinstance(end_layer_index, GraphPath):
                path = end_layer_index
                end_layer_index = end_layer_index.s0
                added_layer = None
            elif isinstance(end_layer_index, tuple):
                added_layer = end_layer_index[1]
                end_layer_index = end_layer_index[0]
                path = None
            else:
                added_layer = None
                path = None
            for layer in self.base_net[start_layer_index: end_layer_index]:
                x = layer(x)
            if added_layer:
                y = added_layer(x)
            else:
                y = x
            if path:
                sub = getattr(self.base_net[end_layer_index], path.name)
                for layer in sub[:path.s1]:
                    x = layer(x)
                y = x
                for layer in sub[path.s1:]:
                    x = layer(x)
                end_layer_index += 1
            start_layer_index = end_layer_index
            confidence, location =  self.compute_header(header_index, y)
            header_index += 1
            confidences.append(confidence)
            locations.append(location)

        for layer in self.base_net[end_layer_index:]:
            x = layer(x)
        
        # 추가 레이어에서 위치 헤더와 분류 헤더 계산
        for layer in self.extras:
            x = layer(x)
            confidence, location = self.compute_header(header_index, x)
            header_index += 1
            confidences.append(confidence)
            locations.append(location)
        
        confidences = torch.cat(confidences, 1)
        locations = torch.cat(locations, 1)

        # 테스트 케이스라면 소프트맥스로 계산한 신뢰도와 박스의 값을 리턴
        # 아니면 신뢰도와 위치 값 리턴
        if self.is_test:
            confidences = F.softmax(confidences, dim=2)
            boxes = box_utils.convert_locations_to_boxes(
                locations, self.priors, self.config.center_variance, self.config.size_variance
            )
            boxes = box_utils.center_form_to_corner_form(boxes)
            return confidences, boxes
        else:
            return confidences, locations
        
    # 클래스 신뢰도와 위치 계산 함수
    def compute_header(self, i, x):
        confidence = self.classification_headers[i](x)
        confidence = confidence.permute(0, 2, 3, 1).contiguous()
        confidence = confidence.view(confidence.size(0), -1, self.num_classes)

        location = self.regression_headers[i](x)
        location = location.permute(0, 2, 3, 1).contiguous()
        location = location.view(location.size(0), -1, 4)

        return confidence, location

    # 기본 네트워크 (백본) 에서 시작하는 함수
    def init_from_base_net(self, model):
        self.base_net.load_state_dict(
            torch.load(model, map_location=lambda storage, loc: storage),
            strict=True
        )
        self.source_layer_add_ons.apply(_xavier_init_)
        self.extras.apply(_xavier_init_)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xvaier_init_)

    # 미리 훈련된 SSD 네트워크에서 시작하는 함수
    def init_from_pretrained_ssd(self, model):
        state_dict = torch.load(model, map_location=lambda storage, loc: storage)
        state_dict = {k: v for k, v in state_dict.items() if not (k.startswith("classification_headers") or k.startswish("regression_headers"))}
        model_dict = self.state_dict()
        model_dict.update(state_dict)
        self.load_state_dict(model_dict)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

    # SSD 초기화 함수
    def init(self):
        self.base_net.apply(_xavier_init_)
        self.source_layer_add_ons.apply(_xavier_init_)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

    # 학습된 SSD의 가중치를 모델 파일로 저장
    def save(self, model_path):
        torch.save(self.state_dict(), model_path)