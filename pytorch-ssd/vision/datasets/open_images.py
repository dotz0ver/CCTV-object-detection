import numpy as np
import pathlib
import cv2
import pandas as pd
import copy
import os
import logging

class OpenImagesDataset:
    # 이미지 경로와 이미지 전처리, 데이터의 정보 수집 등 이미지 데이터세트 초기화
    def __init__(self, root,
                transform=None, target_transform=None,
                dataset_type="train", balance_data=False):
        self.root = pathlib.Path(root)
        self.transform = transform
        self.target_transform = target_transform
        self.dataset_type = dataset_type.lower()

        self.data, self.class_names, self.class_dict = self.read_data()
        self.balance_data = balance_data
        self.min_image_num = -1
        if self.balance_data:
            self.data = self._balance_data()
        self.ids = [info['image_id'] for info in self.data]

        self.class_stat = None

    # 데이터의 인덱스를 입력받고 데이터를 읽고 인덱스의 이미지와 클래스 정보 반환
    def _getitem(self, index):
        image_info = self.data[index]
        image = self._read_image(image_info['image_id'])
        # 데이터세트 보존을 위한 박스 객체 복사
        boxes = copy.copy(image_info['boxes'])
        boxes[:, 0] *= image.shape[1]
        boxes[:, 1] *= image.shape[0]
        boxes[:, 2] *= image.shape[1]
        boxes[:, 3] *= image.shape[0]
        # 데이터세트 보존을 위한 라벨 객체 복사
        labels = copy.copy(image_info['labels'])
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        return image_info['image_id'], image, boxes, labels
    
    def __getitem__(self, index):
        _, image, boxes, labels = self._getitem(index)
        return image, boxes, labels
    
    # 데이터의 인덱스를 입력받고 라벨 정보를 반환
    def get_annotatiion(self, index):
        image_id, image, boxes, labels = self._getitem(index)
        is_difficult = np.zeros(boxes.shape[0], dtype=np.uint8)
        return image_id, (boxes, labels, is_difficult)
    
    # 데이터의 인덱스를 입력받고 이미지를 반환
    def get_image(self, index):
        image_info = self.data[index]
        image = self._read_image(image_info['image_id'])
        if self.transform:
            image, _ = self.transform(image)
        return image
    
    # 애너테이션 파일에서 이미지 정보, 박스, 라벨 정보를 읽음
    def _read_data(self):
        annotation_file = f"{self.root}/sub-{self.dataset_type}-annotations-bbox.csv"
        logging.info(f'loading annotations from: {annotation_file}')
        annotations = pd.read_csv(annotation_file)
        logging.info(f'annotations loaded from: {annotation_file}')
        class_names = ['BACKGROUND'] + sorted(list(annotations['ClassName'].unique()))
        class_dict = {class_name: i for i, class_name in enumerate(class_names)}
        data = []
        for image_id, group in annotations.groupby("ImageID"):
            img_path = os.path.join(self.root, self.dataset_type, image_id + '.jpg')
            if os.path.isfile(img_path) is False:
                logging.error(
                    f'missing ImageID {image_id}.jpg - dropping from annotations'
                )
                continue
            boxes = group.loc[:, ["XMin", "YMin", "XMax", "YMax"]].values.astype(
                np.float32
            )
            # 교차 엔트로피 함수를 만족하기 위해 라벨을 64비트로 만든다.
            labels = np.array(
                [class_dict[name] for name in group["ClassName"]], dtype='int64'
            )

            data.append({'image_id': image_id, 'boxes': boxes, 'labels': labels})
        print('num images: {:d}'.format(len(data)))
        return data, class_names, class_dict
    
    # 파일에서 이미지를 읽는 함수
    def _read_image(self, image_id):
        image_file = self.root / self.dataset_type / f"{image_id}.jpg"
        image = cv2.imread(str(image_file))
        if image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    # 데이터세트에 클래스별 숫자 밸런스를 맞춰줌
    def _balance_data(self):
        logging.info('balancing data')
        label_image_indexes = [set() for _ in range(len(self.class_names))]
        for i, image in enumerate(self.data):
            for label_id in image['labels']:
                label_image_indexes[label_id].add(i)
        label_stat = [len(s) for s in label_image_indexes]
        self.min_image_num = min(label_stat[1:])
        sample_image_indexes = set()
        for image_indexes in label_image_indexes[1:]:
            image_indexes = np.array(list(image_indexes))
            sub = np.random.permutation(image_indexes)[:self.min_image_num]
            sample_image_indexes.update(sub)
        sample_data = [self.data[i] for i in sample_image_indexes]
        return sample_data