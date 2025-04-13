import torch
import torch.nn.functional as F
from detectron2.engine import DefaultPredictor
from detectron2.structures import Instances
from detectron2.utils.visualizer import GenericMask
from detectron2.structures import BoxMode
from detectron2.data.datasets import load_coco_json
from detectron2.structures import BoxMode
import torch
from detectron2.data import DatasetCatalog, MetadataCatalog
import numpy as np
from tqdm import tqdm
import torch
from detectron2.data import DatasetCatalog, MetadataCatalog
import numpy as np
from tqdm import tqdm
import cv2


class LogitsHook:
    def __init__(self):
        self.reset()
    
    def __call__(self, module, input, output):
        # 로짓을 저장
        self.logits.append(output.detach())
    
    def reset(self):
        self.logits = []

class BoxesHook:
    def __init__(self):
        self.reset()
    
    def __call__(self, module, input, output):
        # 입력 proposal_boxes를 저장
        if isinstance(input[0], list) and len(input[0]) > 0:
            proposal_boxes = [x.proposal_boxes.tensor for x in input[0]]
            if proposal_boxes:
                self.boxes.append(torch.cat(proposal_boxes, dim=0))
    
    def reset(self):
        self.boxes = []

class ConformalPredictor(DefaultPredictor):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model.eval()
        
        # 로짓을 위한 훅
        self.logits_hook = LogitsHook()
        self.model.roi_heads.box_predictor.cls_score.register_forward_hook(self.logits_hook)
        self.qhat_k = []

        
        # 명시적 디버깅 메시지 추가
        print("ConformalPredictor initialized with hooks")
    
    def __call__(self, original_image):
        # 훅 데이터 초기화
        self.logits_hook.reset()
        
        # 원래 예측 수행
        outputs = super().__call__(original_image)
        
        # 훅에서 로짓을 얻지 못한 경우
        if not self.logits_hook.logits:
            print("WARNING: No logits captured by hook")
            return outputs
        
        # 클래스 확률 계산
        all_logits = torch.cat(self.logits_hook.logits, dim=0)
        all_probs = F.softmax(all_logits, dim=1)
        
        # 인스턴스에 클래스 확률 직접 추가
        if "instances" in outputs and len(outputs["instances"]) > 0:
            instances = outputs["instances"]
            scores = instances.scores
            pred_classes = instances.pred_classes

            matched_probs = []

            all_max_probs, all_max_classes = torch.max(all_probs, dim=1)

            for i in range(len(pred_classes)):
                score = scores[i]
                pred_class = pred_classes[i]

                # softmax 결과에서 top-1 class가 pred_class이고, score가 일치하는 index 찾기
                candidates = (all_max_classes == pred_class) & torch.isclose(all_max_probs, score, atol=1e-4)
                idxs = torch.nonzero(candidates)  # 중요: .squeeze() 제거
                
                if idxs.numel() > 0:
                    # 첫 번째 매칭된 인덱스 사용
                    if idxs.dim() == 2:  # [n, 1] 형태의 텐서
                        idx_value = idxs[0, 0].item()
                    else:  # 단일 값인 경우
                        idx_value = idxs.item()
                    
                    matched_probs.append(all_probs[idx_value])
                else:
                    # 매칭되는 것이 없으면 합성 분포 생성
                    synthetic_prob = torch.zeros_like(all_probs[0])
                    
                    # 정확한 클래스 위치에 점수 할당 (배경 클래스는 인덱스 0)
                    synthetic_prob[pred_class + 1] = score.item()  # 예측 클래스에 점수 할당
                    synthetic_prob[0] = 1.0 - score.item()  # 배경 클래스에 나머지 할당
                    
                    matched_probs.append(synthetic_prob)
            
            if matched_probs:
                matched_probs_tensor = torch.stack(matched_probs)
                instances.class_probs = matched_probs_tensor
                #print(f"Added matched class_probs of shape {matched_probs_tensor.shape}")        
        
        return outputs


def build_conformal_predictor(cfg):
    return ConformalPredictor(cfg)

def coco_label_from_index(indices):
    """
    주어진 1차원 텐서, 리스트 또는 단일 정수를 COCO 라벨 텍스트 또는 리스트로 변환합니다.
    각 인덱스는 0~79 사이의 COCO 객체 클래스여야 합니다.
    """
    coco_labels = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
        "truck", "boat", "traffic light", "fire hydrant", "stop sign",
        "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
        "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
        "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
        "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
        "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv",
        "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
        "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush", "background"
    ]

    # Tensor → list로 변환
    if isinstance(indices, torch.Tensor):
        indices = indices.tolist()

    # 단일 정수 입력
    if isinstance(indices, int):
        return coco_labels[indices] if 0 <= indices < len(coco_labels) else "Invalid index"

    # 리스트 입력
    return [coco_labels[i] if 0 <= i < len(coco_labels) else "Invalid index" for i in indices]




def compute_iou(boxA, boxB):
    # standard IoU function
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)


def get_conformal_predictor(model, dataset, alpha=0.1):
    '''
    주어진 모델과 dataset에 대해서, class-wise quantile을 계산합니다.
    
    Args:
        model: 컨포멀 예측기 모델
        dataset: 데이터셋 이름 또는 데이터셋 객체
        alpha: 목표 오류율 (기본값: 0.1, 즉 90% 신뢰도)
    
    Returns:
        class_qhats: 클래스별 quantile 값 딕셔너리
    '''
    
    # 데이터셋 로드
    if isinstance(dataset, str):
        dataset_dicts = DatasetCatalog.get(dataset)
        metadata = MetadataCatalog.get(dataset)
    else:
        dataset_dicts = dataset
    
    print(f"Computing calibration scores for {len(dataset_dicts)} images...")
    
    # 클래스별 non-conformity score를 저장할 딕셔너리
    class_scores = {}
    for i in range(80):  # COCO의 80개 클래스에 대해
        class_scores[i] = []
    
    # 각 이미지에 대해 예측 수행
    for i, data in enumerate(tqdm(dataset_dicts)):
        if i % 100 == 0:
            print(f"Processing image {i}/{len(dataset_dicts)}")
        
        # 이미지 로드
        img_path = data["file_name"]
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        # 예측
        outputs = model(img)
        instances = outputs["instances"]
        
        if len(instances) == 0:
            continue
        
        # 예측 결과 추출
        boxes = instances.pred_boxes.tensor.cpu().numpy()
        scores = instances.scores.cpu().numpy()
        classes = instances.pred_classes.cpu().numpy()
        class_probs = instances.class_probs.cpu().numpy()
        
        # Ground truth 데이터 추출
        gt_boxes = np.array([ann["bbox"] for ann in data["annotations"]])
        if len(gt_boxes) == 0:
            continue
            
        # XYWH → XYXY 변환 (필요한 경우)
        from detectron2.structures.boxes import BoxMode
        gt_boxes = BoxMode.convert(gt_boxes, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
        gt_classes = np.array([ann["category_id"] - 1 for ann in data["annotations"]])  # COCO는 1부터 시작하므로 -1
        
        # 각 gt 박스에 대해 가장 IoU가 높은 예측 박스 찾기
        from detectron2.structures.boxes import pairwise_iou
        from detectron2.structures import Boxes
        
        pred_boxes_obj = Boxes(torch.tensor(boxes))
        gt_boxes_obj = Boxes(torch.tensor(gt_boxes))
        
        ious = pairwise_iou(gt_boxes_obj, pred_boxes_obj).cpu().numpy()
        
        # 클래스별로 non-conformity score 계산
        for gt_idx, gt_class in enumerate(gt_classes):
            # 해당 gt와 매칭되는 가장 높은 IoU를 가진 예측 박스 찾기
            if ious[gt_idx].size == 0:
                continue
                
            max_iou_idx = np.argmax(ious[gt_idx])
            max_iou = ious[gt_idx][max_iou_idx]
            
            # IoU 임계값 (ex: 0.5)보다 높은 경우만 처리
            if max_iou >= 0.5:
                pred_class = classes[max_iou_idx]
                pred_score = scores[max_iou_idx]
                pred_probs = class_probs[max_iou_idx]
                
                # 올바른 클래스를 예측한 경우
                if pred_class == gt_class:
                    # Non-conformity score = 1 - P(y | x)
                    # 클래스별 확률 분포에서 정답 클래스의 확률 사용
                    true_class_prob = pred_probs[gt_class + 1]  # 배경 클래스(0) 때문에 +1
                    nonconf_score = 1.0 - true_class_prob
                    
                    # 클래스별 점수 저장
                    class_scores[gt_class].append(nonconf_score)
    
    # 클래스별 quantile 계산
    class_qhats = {}
    for class_id, scores in class_scores.items():
        if len(scores) > 0:
            # 각 클래스의 보정 수행 (n+1)/n 점수를 사용
            n = len(scores)
            q_level = np.ceil((n+1) * (1-alpha)) / n
            
            # Quantile 계산
            qhat = np.quantile(scores, q_level) if len(scores) > 0 else 1.0
            class_qhats[class_id] = qhat
            
            label = coco_label_from_index(class_id)
            print(f"Class {class_id} ({label}): {len(scores)} samples, qhat = {qhat:.4f}")
        else:
            class_qhats[class_id] = 1.0  # 데이터가 없는 경우 기본값
            label = coco_label_from_index(class_id)
            print(f"Class {class_id} ({label}): 0 samples, qhat = 1.0000")
    
    return class_qhats

'''

image_root = "/home/datasets/coco/val2017"
json_file = "/home/datasets/coco/annotations/instances_val2017.json"

val_dataset_dicts = load_coco_json(json_file, image_root, dataset_name="coco_val2017")
calibration_scores = [] # calibration dataset의 score. adaptive set이 아님
cnt = 0

for data in val_dataset_dicts:
    if cnt % 500 == 0:
        print(f"processing data is : {data["file_name"]}, iter : {cnt}")
    cnt += 1
    
    img = cv2.imread(data["file_name"])
    outputs = predictor(img)

    pred_boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
    pred_scores = outputs["instances"].scores.cpu().numpy()
    pred_classes = outputs["instances"].pred_classes.cpu().numpy()

    gt_boxes = [BoxMode.convert(ann["bbox"], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS) for ann in data["annotations"]]
    gt_classes = [ann["category_id"] for ann in data["annotations"]]

    matched_gt = set()  # 이미 매칭된 ground truth box의 인덱스
    for i, gt_box in enumerate(gt_boxes):
        best_iou = 0
        best_idx = -1
        for j, pred_box in enumerate(pred_boxes):
            if j in matched_gt:
                continue
            iou = compute_iou(gt_box, pred_box)
            if iou >= 0.5 and iou > best_iou:
                best_iou = iou
                best_idx = j
        if best_idx != -1:
            matched_gt.add(best_idx)
            score = pred_scores[best_idx]
            # Non-conformity score = 1 - P(y | x)
            calibration_scores.append(1 - score)
print(len(calibration_scores))
calibration_scores = np.array(calibration_scores)
print(calibration_scores.shape)
print(calibration_scores[:-10])

sorted_score = np.sort(calibration_scores)
alpha = 0.05
n = len(sorted_score)
quantile_index = int(np.ceil((1 - alpha) * (n + 1))) - 1 # 0-based index

#quantile_index = int(np.ceil((n + 1) * (1 - alpha)/n))
q_hat = sorted_score[quantile_index]
print(f"q_hat: {q_hat}, n = {n}, quantile_index = {quantile_index}")
print(sorted_score[:10], n, quantile_index, q_hat)


'''





# 예제
if __name__ == "__main__":
    import torch
    indices = torch.asarray([0, 1, 2, 3, 4, 5, 6, 77, 78, 79,80])
    labels = coco_label_from_index(indices)
    print(f"Indices: {indices.tolist()}")
    print(f"Labels: {labels}")