import torch
import torch.nn.functional as F
from detectron2.engine import DefaultPredictor
from detectron2.structures import Instances
from detectron2.utils.visualizer import GenericMask
from detectron2.structures import BoxMode
from detectron2.data.datasets import load_coco_json
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from utilities.label_utils import coco_label_from_index
import numpy as np
from tqdm import tqdm
import cv2
import json
import os
from pycocotools.coco import COCO


class LogitsHook:
    '''
    Detectron 모델이 logit을 출력할 수 없으므로, 내부에 hook을 걸어서 logit값을 별도로 추출해야 함
    '''
    def __init__(self):
        self.reset()
    
    def __call__(self, module, input, output):
        # 로짓을 저장
        self.logits.append(output.detach())
    
    def reset(self):
        self.logits = []

class BoxesHook:
    '''
    아직은 사용하지 않음
    '''
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
    '''
    예측기
    '''
    def __init__(self, cfg):
        super().__init__(cfg)     # 모델은 defaultpredictor를 상속받음
        self.model.eval()         # 학습이 아닌 평가모델이므로 인스턴스 생성시 기본으로 eval mode
        
        # 로짓을 위한 hook 생성
        self.logits_hook = LogitsHook()
        self.model.roi_heads.box_predictor.cls_score.register_forward_hook(self.logits_hook)
        self.qhat_k = [] # 각 클래스별 qhat
    
    def __call__(self, original_image):
        '''
        inference. 함수가 호출되면
        '''
        self.logits_hook.reset() # hook을 리셋함
        ### Forward
        outputs = super().__call__(original_image) # 모델에 이미지를 입력시키고 출력을 반환받음.

        # logits이 capture되지 않은 경우
        if not self.logits_hook.logits:
            print("WARNING: No logits captured by hook")  # 예외처리 - 디버깅용 오류메세지
            return outputs

        # logits을 hook으로부터 받아옴
        all_logits = torch.cat(self.logits_hook.logits, dim=0)  # hook로부터 로짓을 받음
        all_probs = F.softmax(all_logits, dim=1)                # 로짓을 소프트맥스 함수를 통해 확률로 변환함

        # LogitsHook에서 logits을 가져옴
        if "instances" in outputs and len(outputs["instances"]) > 0: #조건 : 모델 출력의 인스턴스가 존재하고, 인스턴스의 길이가 0보다 클 때
            outputs["instances"] = self.assign_class_probabilities(outputs["instances"], all_probs) #출력의 "instances"에 대해서 확률 출력을 덧붙임

        return outputs


    def load_calibration(self, calib_path):
        '''
        load calibration, calibration JSON 파일을 불러옴. 이 Calibration JSON은 valToCal.py에서 생성한 JSON 파일을 사용
        calibration 파일이 없으면, 추론을 하면서 Calibration dataset을 얻기 위한 긴 과정을 진행해 주어야 한다.
        '''
        with open(calib_path, "r") as f:
            data = json.load(f)
        self.classwise_qhat = {int(k): v for k, v in data["classwise_qhat"].items()}
        self.global_qhat = data["global_qhat"]
    
        # 명시적 디버깅 메시지 추가
        print("Loaded calibration data")

    def get_adaptive_prediction_set(self, prob_tensor):
        """
        Adaptive Prediction Set using APS strategy.
        Greedily include classes with highest probabilities until the cumulative probability exceeds 1 - q̂.
        Always includes at least one class.
        """
        sorted_probs, sorted_indices = torch.sort(prob_tensor, descending=True)
        cumulative = 0.0
        selected = []
        for prob, idx in zip(sorted_probs, sorted_indices):
            cumulative += prob.item()
            selected.append(idx.item())
            if cumulative >= 1 - self.global_qhat:
                break
        return selected
    
    def get_max_residuals_for_classes(self, class_list):
        max_residuals = [0.0, 0.0, 0.0, 0.0]
        for cls in class_list:
            if cls in self.classwise_qhat:
                q = self.classwise_qhat[cls]["90%"]
                for j in range(4):
                    max_residuals[j] = max(max_residuals[j], q[j + 1])
        return max_residuals
    
    def expand_box_with_residuals(self, box, residuals):
        x1_min = box[0] - residuals[0]
        y1_min = box[1] - residuals[1]
        x2_max = box[2] + residuals[2]
        y2_max = box[3] + residuals[3]
        print(x1_min, y1_min, x2_max, y2_max)
        return {
            "x1_min": x1_min.item(), "x1_max": box[0].item(),
            "y1_min": y1_min.item(), "y1_max": box[1].item(),
            "x2_min": box[2].item(), "x2_max": x2_max.item(),
            "y2_min": box[3].item(), "y2_max": y2_max.item(),
        }

    def get_predicted_set(self, prob):
        return self.get_adaptive_prediction_set(prob)

    def compute_conformal_region(self, box, labels_in_set):
        max_residuals = self.get_max_residuals_for_classes(labels_in_set)
        return self.expand_box_with_residuals(box, max_residuals)

    def conformalize(self, instances):
        pred_set = []
        conformal_regions = []

        for i in range(len(instances)):
            labels_in_set = self.get_predicted_set(instances.class_probs[i])
            label_names = coco_label_from_index(labels_in_set)
            pred_set.append(label_names)
            if not labels_in_set:
                continue
            region = self.compute_conformal_region(instances.pred_boxes.tensor[i], labels_in_set)
            #print(f"Conformal region box: {region}\n, point box: {instances.pred_boxes.tensor[i]}") # 디버깅용 메세지
            conformal_regions.append(region)

        instances.pred_set = pred_set
        instances.conformal_region = conformal_regions

    def assign_class_probabilities(self, instances, all_probs):
        """
        클래스 확률 분포를 할당함.
        """
        scores = instances.scores
        pred_classes = instances.pred_classes
        matched_probs = []
        all_max_probs, all_max_classes = torch.max(all_probs, dim=1)
        for i, (score, pred_class) in enumerate(zip(scores, pred_classes)):
            # 예측 클래스와 점수가 일치하는 확률 분포 찾기
            candidates = (all_max_classes == pred_class) & torch.isclose(all_max_probs, score, atol=1e-4)
            idxs = torch.nonzero(candidates)
            
            if idxs.numel() > 0:
                # 매칭된 확률 분포 사용
                idx_value = idxs[0, 0].item() if idxs.dim() == 2 else idxs.item()
                matched_probs.append(all_probs[idx_value])
            else:
                synthetic_prob = torch.zeros_like(all_probs[0])
                synthetic_prob[pred_class + 1] = score.item()  # 예측 클래스
                synthetic_prob[0] = 1.0 - score.item()  # 배경 클래스
                matched_probs.append(synthetic_prob)
        
        if matched_probs:
            matched_probs_tensor = torch.stack(matched_probs)
            instances.class_probs = matched_probs_tensor
        
        return instances

def build_conformal_predictor(cfg):
    '''
    conformal prediction를 빌드하는 함수
    '''
    return ConformalPredictor(cfg)

if __name__ == "__main__":
    import os
    import cv2
    from detectron2.config import get_cfg
    from detectron2 import model_zoo
    from detectron2.utils.visualizer import Visualizer
    from detectron2.data import MetadataCatalog

    # Config
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.DEVICE = 'cuda'

    # Load test image
    test_img_path = "/home/datasets/coco/val2017/000000000785.jpg"
    test_json_file = "/home/datasets/coco/annotations/instances_val2017.json"
    image = cv2.imread(test_img_path)

    # Build predictor
    predictor = build_conformal_predictor(cfg)
    predictor.load_calibration("calib.json")  # Load calibration values

    # Predict and conformalize
    outputs = predictor(image)
    predictor.conformalize(outputs["instances"])

    # Visualization
    v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.0)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    # Draw conformal region boxes
    conformal_regions = outputs["instances"].conformal_region
    output_img = out.get_image()[:, :, ::-1].copy()  # <--- 이 줄에 .copy() 추가
    for region in conformal_regions:
        pt1 = (int(region["x1_min"]), int(region["y1_min"]))
        pt2 = (int(region["x2_max"]), int(region["y2_max"]))
        cv2.rectangle(output_img, pt1, pt2, (0, 255, 0), 1)

    # Load ground-truth boxes using pycocotools
    coco = COCO(test_json_file)
    file_name = os.path.basename(test_img_path)
    img_id = None
    for k, v in coco.imgs.items():
        if v["file_name"].endswith(file_name):
            img_id = k
            break
    if img_id is None:
        raise ValueError(f"Image ID not found for {file_name}")
    ann_ids = coco.getAnnIds(imgIds=img_id)
    """
        # Draw ground-truth boxes in red
        for ann in anns:
            bbox = ann["bbox"]  # format: [x, y, width, height]
            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = x1 + int(bbox[2])
            y2 = y1 + int(bbox[3])
            cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 0, 255), 1)
    """
    # Save or display
    cv2.imwrite("output_conformal.jpg", output_img)
    print("Saved visualization to output_conformal.jpg")