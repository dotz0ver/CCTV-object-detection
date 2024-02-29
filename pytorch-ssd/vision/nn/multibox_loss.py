import torch.nn as nn
import torch.nn.functional as F
import torch


from ..utils import box_utils

# 분류, 위치 예측의 두 가지 종류의 손실 고려를 위함
class MultiboxLoss(nn.Module):
    def __init__(self, priors, iou_threshold, neg_pos_ratio,
                 center_variance, size_variance, device):
        """Implement SSD Multibox Loss.

        Basically, Multibox loss combines classification loss
         and Smooth L1 regression loss.
        """
        super(MultiboxLoss, self).__init__()
        self.iou_threshold = iou_threshold # 정답 박스와 예측 박스 간 IoU 임계값
        self.neg_pos_ratio = neg_pos_ratio # 하드 네거티브 마이닝 (부정 예측 : 양성 예측)
        self.center_variance = center_variance # 바운딩 박스 예측 오차
        self.size_variance = size_variance
        self.priors = priors # 모델이 예측해야 하는 초기 바운딩 박스
        self.priors.to(device)

    def forward(self, confidence, predicted_locations, labels, gt_locations):
        """Compute classification loss and smooth l1 loss.

        Args:
            confidence (batch_size, num_priors, num_classes): class predictions.
            locations (batch_size, num_priors, 4): predicted locations.
            labels (batch_size, num_priors): real labels of all the priors.
            boxes (batch_size, num_priors, 4): real boxes corresponding all the priors.
        """
        num_classes = confidence.size(2)
        with torch.no_grad():
            # derived from cross_entropy=sum(log(p))
            loss = -F.log_softmax(confidence, dim=2)[:, :, 0]
            # cross entropy loss 에서 중요하지 않은 배경 class에 대한 예측을 걸러내기 위한 하드 네거티브 마이닝
            mask = box_utils.hard_negative_mining(loss, labels, self.neg_pos_ratio)

        confidence = confidence[mask, :] # 모델이 예측한 클래스별 확률
        # 분류 성능
        classification_loss = F.cross_entropy(confidence.reshape(-1, num_classes), labels[mask], size_average=False)
        pos_mask = labels > 0
        predicted_locations = predicted_locations[pos_mask, :].reshape(-1, 4)
        gt_locations = gt_locations[pos_mask, :].reshape(-1, 4) # 실제 바운딩 박스 위치
        # predicted_locations (예측 위치)와 gt_locations)실제 위치) 간 위치 예측 손실 계산
        smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, size_average=False)
        num_pos = gt_locations.size(0)
        return smooth_l1_loss/num_pos, classification_loss/num_pos