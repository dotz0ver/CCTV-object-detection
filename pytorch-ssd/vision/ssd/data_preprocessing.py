from ..transforms.transforms import *

# 1. 훈련 데이터 변환 적용
class TrainAugmentation:
    def __init__(self, size, mean=0, std=1.0):
        """
        Args:
            size: the size the of final image.
            mean: mean pixel value per channel.
        """
        self.mean = mean
        self.size = size
        # 이미지 데이터 변환
        self.augment = Compose([
            ConvertFromInts(), # 정수 형식을 다른 데이터 타입으로 변환
            PhotometricDistort(), # 이미지에 포토메트릭 변형을 적용하여 조명 등 다양화
            Expand(self.mean), # 이미지 확장, 평균값으로 패딩
            RandomSampleCrop(),
            RandomMirror(), # 랜덤하게 이미지 좌우 반전
            ToPercentCoords(), # 박스의 좌표를 이미지 크기에 대한 상대 좌표로 변환
            Resize(self.size), # 이미지를 저장된 크기로 조절
            SubtractMeans(self.mean), # 이미지 - 평균값
            # lambda로 이미지를 표준편차로 나누고, 텐서로 변환
            lambda img, boxes=None, labels=None: (img / std, boxes, labels),
            ToTensor(),
        ])

    def __call__(self, img, boxes, labels):
        """

        Args:
            img: the output of cv.imread in RGB layout.
            boxes: boundding boxes in the form of (x1, y1, x2, y2).
            labels: labels of boxes.
        """
        return self.augment(img, boxes, labels)

# 2. 테스트 데이터에 대한 변환
class TestTransform:
    def __init__(self, size, mean=0.0, std=1.0):
        self.transform = Compose([
            ToPercentCoords(),
            Resize(size),
            SubtractMeans(mean),
            lambda img, boxes=None, labels=None: (img / std, boxes, labels),
            ToTensor(),
        ])

    def __call__(self, image, boxes, labels):
        return self.transform(image, boxes, labels)

# 3. 모델의 추론을 위한 데이터 변환
class PredictionTransform:
    def __init__(self, size, mean=0.0, std=1.0):
        self.transform = Compose([
            Resize(size),
            SubtractMeans(mean),
            lambda img, boxes=None, labels=None: (img / std, boxes, labels),
            ToTensor()
        ])

    def __call__(self, image):
        image, _, _ = self.transform(image)
        return image