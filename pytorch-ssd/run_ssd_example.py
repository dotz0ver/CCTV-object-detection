#
# SSD 모델 추론 파이썬 코드
#

#1. 시스템, opencv, 그리고 하위 디렉토리 vision에 있는 ssd 패키지 임포트
from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.utils.misc import Timer
import cv2
import sys

#2. 실행 인자 파서로 네트워크, 모델 폴더, 라벨 폴더, 이미지 폴더 세팅
if len(sys.argv) < 5:
    print('Usage: python run_ssd_example.py <net type>  <model path> <label path> <image path>')
    sys.exit(0)
net_type = sys.argv[1]
model_path = sys.argv[2]
label_path = sys.argv[3]
image_path = sys.argv[4]

#3. 라벨 파일을 읽어 클래스 이름들 세팅
class_names = [name.strip() for name in open(label_path).readlines()]

#4. 모델 파일을 읽어 모델 로딩
if net_type == 'vgg16-ssd':
    net = create_vgg_ssd(len(class_names), is_test=True)
elif net_type == 'mb1-ssd':
    net = create_mobilenetv1_ssd(len(class_names), is_test=True)
elif net_type == 'mb1-ssd-lite':
    net = create_mobilenetv1_ssd_lite(len(class_names), is_test=True)
elif net_type == 'mb2-ssd-lite':
    net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)
elif net_type == 'sq-ssd-lite':
    net = create_squeezenet_ssd_lite(len(class_names), is_test=True)
else:
    print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
    sys.exit(1)
net.load(model_path)

#4. 네트워크 지정
if net_type == 'vgg16-ssd':
    predictor = create_vgg_ssd_predictor(net, candidate_size=200)
elif net_type == 'mb1-ssd':
    predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)
elif net_type == 'mb1-ssd-lite':
    predictor = create_mobilenetv1_ssd_lite_predictor(net, candidate_size=200)
elif net_type == 'mb2-ssd-lite':
    predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200)
elif net_type == 'sq-ssd-lite':
    predictor = create_squeezenet_ssd_lite_predictor(net, candidate_size=200)
else:
    predictor = create_vgg_ssd_predictor(net, candidate_size=200)

#5. 이미지 파일을 읽고 추론을 위해 변환
orig_image = cv2.imread(image_path)
image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)

#6. 모델에 이미지 파일을 입력하여 추론
boxes, labels, probs = predictor.predict(image, 10, 0.4)

#7. 추론 결과인 박스와 클래스 명을 이미지에 표시
for i in range(boxes.size(0)):
    box = boxes[i, :]

    cv2.rectangle(orig_image, (int(box[0]), int(box[1]), int(box[2]), int(box[3])), (255, 255, 0), 4)
    #label = f"""{voc_dataset.class_names[labels[i]]}: {probs[i]:.2f}"""
    label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
    Data = "classID:{} box({}, {}, {}, {}) conf:{}".format(labels[i], int(box[0]),int(box[1]),int(box[2]),int(box[3]), probs[i])
    print(Data)
    cv2.putText(orig_image, label,
                (int(box[0]) + 20, int(box[1]) + 40),    
                cv2.FONT_HERSHEY_SIMPLEX,
                1,  # font scale
                (255, 0, 255),
                2)  # line type
    
#5. 결과 이미지 파일 저장   
path = "run_ssd_example_output.jpg"
cv2.imwrite(path, orig_image)
print(f"Found {len(probs)} objects. The output image is {path}")