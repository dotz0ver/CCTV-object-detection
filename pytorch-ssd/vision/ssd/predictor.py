# 객체 검출 시스템

import torch

from ..utils import box_utils
from .data_preprocessing import PredictionTransform
from ..utils.misc import Timer # 추론 시간 측정

# 주어진 신경망('net')을 사용하여 객체 검출
class Predictor:
    def __init__(self, net, size, mean=0.0, std=1.0, nms_method=None,
                 iou_threshold=0.45, filter_threshold=0.01, candidate_size=200, sigma=0.5, device=None):
        self.net = net
        self.transform = PredictionTransform(size, mean, std)
        self.iou_threshold = iou_threshold
        self.filter_threshold = filter_threshold
        self.candidate_size = candidate_size
        self.nms_method = nms_method

        self.sigma = sigma
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.net.to(self.device)
        self.net.eval()

        self.timer = Timer()

    # 입력 이미지, 반환할 최고의 K개 예측 (기본값 -1로 모든 예측 반환), 신뢰도 점수 필터링을 위한 임계값
    def predict(self, image, top_k=-1, prob_threshold=None):
        # 입력 이미지 크기를 가져와 전처리 수행 후 텐서로 변환
        cpu_device = torch.device("cpu")
        height, width, _ = image.shape
        image = self.transform(image)
        #print(image)
        images = image.unsqueeze(0)
        images = images.to(self.device) # 이미지를 모델의 디바이스로 이동
        # 추론 시간 측정을 위해 타이머 시작, 신경망을 통해 순전파 수행
        with torch.no_grad():
            self.timer.start()
            scores, boxes = self.net.forward(images)
            print("Inference time: ", self.timer.end())
        # 예측된 결과 중 첫 번째 이미지의 박스와 점수를 가져옴
        boxes = boxes[0]
        scores = scores[0]
        if not prob_threshold:
            prob_threshold = self.filter_threshold # 기본값
        # this version of nms is slower on GPU, so we move data to CPU.
        # 후처리 수행을 위해 결과를 CPU 디바이스로 이동
        boxes = boxes.to(cpu_device)
        scores = scores.to(cpu_device)
        # 클래스 별로 후처리 수행
        # 각 클래스에 대해 신뢰도 점수를 가져오고, 지정된 임계값보다 높은 점수를 가진 예측 선택
        # 선택된 예측에 대한 박스와 점수를 함께 묶어 리스트에 추가
        picked_box_probs = []
        picked_labels = []
        for class_index in range(1, scores.size(1)):
            probs = scores[:, class_index]
            mask = probs > prob_threshold
            probs = probs[mask]
            if probs.size(0) == 0:
                continue
            subset_boxes = boxes[mask, :]
            box_probs = torch.cat([subset_boxes, probs.reshape(-1, 1)], dim=1)
            box_probs = box_utils.nms(box_probs, self.nms_method,
                                      score_threshold=prob_threshold,
                                      iou_threshold=self.iou_threshold,
                                      sigma=self.sigma,
                                      top_k=top_k,
                                      candidate_size=self.candidate_size)
            picked_box_probs.append(box_probs)
            picked_labels.extend([class_index] * box_probs.size(0))
        # 선택된 박스가 없다면 빈 텐서 반환
        if not picked_box_probs:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])
        # 최종 선택된 박스를 하나의 텐서로 합치고, 원본 이미지에 맞게 조정
        picked_box_probs = torch.cat(picked_box_probs)
        picked_box_probs[:, 0] *= width
        picked_box_probs[:, 1] *= height
        picked_box_probs[:, 2] *= width
        picked_box_probs[:, 3] *= height
        # 선택된 바운딩 박스, 해당 레이블, 신뢰도 점수
        return picked_box_probs[:, :4], torch.tensor(picked_labels), picked_box_probs[:, 4]