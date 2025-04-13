import cppredictor
import torch
import cv2
import numpy as np
import json

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import GenericMask
from detectron2.structures import BoxMode
from detectron2.data.datasets import load_coco_json
from detectron2.structures import BoxMode


### This script is used to evaluate the calibration of a model on the COCO validation dataset.
# input : image_root, json_file of COCO2017
# run : 
# compute_classwise_quantiles : compute the quantiles for boundingbox
# compute_global_quantile : compute the global quantile , for class prediction



class CalibrationEvaluator:
    def __init__(self, image_root, json_file, predictor):
        self.image_root = image_root
        self.json_file = json_file
        self.predictor = predictor
        self.val_dataset_dicts = self.load_dataset()

    def load_dataset(self):
        return load_coco_json(self.json_file, self.image_root, dataset_name="coco_val2017")

    @staticmethod
    def compute_iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea)

    def compute_residuals(self, pred_box, gt_box):
        return [
            abs(pred_box[0] - gt_box[0]),
            abs(pred_box[1] - gt_box[1]),
            abs(pred_box[2] - gt_box[2]),
            abs(pred_box[3] - gt_box[3]),
        ]

    def evaluate_image(self, data):
        img = cv2.imread(data["file_name"])
        outputs = self.predictor(img)

        pred_boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
        pred_scores = outputs["instances"].scores.cpu().numpy()
        pred_classes = outputs["instances"].pred_classes.cpu().numpy()

        gt_boxes = [BoxMode.convert(ann["bbox"], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS) for ann in data["annotations"]]
        gt_classes = [ann["category_id"] for ann in data["annotations"]]

        matched_gt = set()
        classwise_results = {}

        for i, gt_box in enumerate(gt_boxes):
            best_iou = 0
            best_idx = -1
            for j, pred_box in enumerate(pred_boxes):
                if j in matched_gt:
                    continue
                iou = self.compute_iou(gt_box, pred_box)
                if iou >= 0.2 and iou > best_iou:
                    best_iou = iou
                    best_idx = j
            if best_idx != -1:
                matched_gt.add(best_idx)
                score = pred_scores[best_idx]
                cls = pred_classes[best_idx]
                nonconf_score = 1 - score
                residuals = self.compute_residuals(pred_boxes[best_idx], gt_box)
                if cls not in classwise_results:
                    classwise_results[cls] = []
                classwise_results[cls].append((nonconf_score, *residuals))

        return classwise_results

    def compute_classwise_quantiles(self, aggregated_results, alpha=0.1):
        quantile_results = {}
        for cls, values in aggregated_results.items():
            values = np.array(values)  # shape: (N, 5)
            if len(values) == 0:
                continue
            quantiles = np.quantile(values, q=1-alpha, axis=0, method='higher')
            quantile_results[cls] = {
                f"{int((1 - alpha) * 100)}%": quantiles.tolist()
            }
        return quantile_results

    def compute_global_quantile(self, aggregated_results, alpha=0.1):
        all_nonconf_scores = []
        
        for values in aggregated_results.values():
            all_nonconf_scores.extend([v[0] for v in values])  # Extract nonconf scores only
        if not all_nonconf_scores:
            return None
        
        return float(np.quantile(all_nonconf_scores, 1 - alpha, method='higher'))

    def save_results(self, classwise_qhat, global_qhat, save_path):
        output = {
            "classwise_qhat": {str(k): v for k, v in classwise_qhat.items()},
            "global_qhat": global_qhat
        }
        with open(save_path, "w") as f:
            json.dump(output, f, indent=2)

    def load_results(self, load_path):
        with open(load_path, "r") as f:
            loaded_scores = json.load(f)
        return {int(k): v for k, v in loaded_scores.items()}

    def run(self):
        cnt = 0
        aggregated_results = {}

        #for data in self.val_dataset_dicts[:500]: # this is for test
        for data in self.val_dataset_dicts:
            if cnt % 500 == 0:
                print(f"processing data is : {data['file_name']}, iter : {cnt}")
            cnt += 1

            image_results = self.evaluate_image(data)
            for cls, scores in image_results.items():
                if cls not in aggregated_results:
                    aggregated_results[cls] = []
                aggregated_results[cls].extend(scores)

        return aggregated_results
    


if __name__ == "__main__":
    image_root = "/home/datasets/coco/val2017"
    json_file = "/home/datasets/coco/annotations/instances_val2017.json"

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6
    cfg.MODEL.DEVICE = 'cuda'

    predictor = cppredictor.ConformalPredictor(cfg)

    evaluator = CalibrationEvaluator(image_root, json_file, predictor)
    calibration_scores = evaluator.run()
    # Save the calibration scores to a file
    
    print(len(calibration_scores))

    alpha = 0.1

    classwise_qhat = evaluator.compute_classwise_quantiles(calibration_scores, alpha = alpha)
    global_qhat = evaluator.compute_global_quantile(calibration_scores, alpha = alpha)
    print(len(classwise_qhat), global_qhat)
    #evaluator.save_results(classwise_qhat, global_qhat, "calib.json")
    evaluator.save_results(calibration_scores, None, "calib_scores.json")