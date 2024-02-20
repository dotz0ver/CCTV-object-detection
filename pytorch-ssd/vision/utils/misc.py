import time
import torch

# 입력된 문자열이 'true' or '1'인 경우 True
def str2bool(s):
    return s.lower() in ('true', '1')

# 코드 실행 시간 측정
class Timer:
    def __init__(self):
        self.clock = {}

    # 시작
    def start(self, key="default"):
        self.clock[key] = time.time()

    # 끝. 경과 시간 반환
    def end(self, key="default"):
        if key not in self.clock:
            raise Exception(f"{key} is not in the clock.")
        interval = time.time() - self.clock[key]
        del self.clock[key]
        return interval
        
# 학습 중인 모델의 상태를 체크포인트 파일로 저장하는 함수
def save_checkpoint(epoch, net_state_dict, optimizer_state_dict, best_score, checkpoint_path, model_path):
    torch.save({
        'epoch': epoch,
        'model': net_state_dict,
        'optimizer': optimizer_state_dict,
        'best_score': best_score
    }, checkpoint_path)
    torch.save(net_state_dict, model_path)
        
# 체크포인트 파일에서 저장된 모델 상태 로드 
def load_checkpoint(checkpoint_path):
    return torch.load(checkpoint_path)

# 파라미터 업데이트 막기 (사전 훈련된 모델에서 특정 층의 가중치 고정시킬 때)
def freeze_net_layers(net):
    for param in net.parameters():
        param.requires_grad = False

# 라벨들을 파일로 저장
def store_labels(path, labels):
    with open(path, "w") as f:
        f.write("\n".join(labels))