
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


"""
This script is used to generate a calibration dataset from a COCO-style dataset and a pre-trained object detection model. 
It evaluates the model's predictions on the dataset and computes calibration metrics such as residuals and non-conformity scores.
Classes:
    DatasetLoader:
        A utility class to load a COCO-style dataset for calibration purposes.
    BoxCalculations:
        A utility class for performing various box-related calculations, such as IoU, residuals, and box format conversions.
    CalibrationMetrics:
        A utility class for computing and saving calibration metrics, including class-wise and global quantiles.
    CalibrationRunner:
        A class to evaluate a dataset using a predictor and compute aggregated calibration results.
Methods:
    DatasetLoader.__init__(image_root, json_file):
        Initializes the DatasetLoader with the image root directory and the COCO JSON annotation file.
    DatasetLoader.load_dataset():
        Loads the COCO validation dataset as a calibration dataset.
    BoxCalculations.compute_iou(boxA, boxB):
        Computes the Intersection over Union (IoU) between two bounding boxes.
    BoxCalculations.compute_residuals(pred_box, gt_box):
        Computes the residuals (absolute differences) between predicted and ground truth bounding boxes.
    CalibrationMetrics.compute_classwise_quantiles(aggregated_results, alpha=0.1):
        Computes class-wise quantiles for the given aggregated results.
    CalibrationMetrics.compute_global_quantile(aggregated_results, alpha=0.1):
        Computes a global quantile across all classes for the given aggregated results.
    CalibrationMetrics.save_results(classwise_qhat, global_qhat, path):
        Saves the computed calibration metrics (class-wise and global quantiles) to a JSON file.
    CalibrationMetrics.load_results(path):
        Loads calibration metrics from a JSON file.
    CalibrationRunner.__init__(dataset_dicts, predictor):
        Initializes the CalibrationRunner with the dataset and the predictor.
    CalibrationRunner.evaluate_image(data):
        Evaluates a single image by computing residuals and non-conformity scores for predictions with IoU >= 0.2.
    CalibrationRunner.run():
        Iterates through the dataset, evaluates each image, and aggregates the results.
Main Script:
    The script initializes the dataset loader, predictor, and calibration runner, and computes calibration metrics. 
    The results are saved to a JSON file for further analysis.

"""

class DatasetLoader:
    def __init__(self, image_root, json_file):
        self.image_root = image_root
        self.json_file = json_file

    def load_dataset(self):
        """
        load the COCO validation dataset as calibration dataset
        COCO validation dataset을 Calibration dataset으로 사용하기 위해 Load 함
        """
        return load_coco_json(self.json_file, self.image_root, dataset_name="coco_val2017") # return load coco dataset from the path
    
class BoxCalculations:
    """
    compute_iou(boxA, boxB): 두 박스 간의 IoU 계산
    compute_residuals(pred_box, gt_box): 예측 박스와 실제 박스 간의 잔차 계산
    convert_box_format(box, from_format, to_format): 박스 포맷 변환 (XYWH_ABS ↔ XYXY_ABS)
    is_valid_box(box): 박스가 유효한지 검사 (너비/높이 > 0, 좌표 순서 맞는지 등)
    clip_box(box, image_width, image_height): 이미지 경계를 넘어가는 박스 클리핑
    """
    @staticmethod
    def compute_iou(boxA, boxB):
        """
        input : boxA, boxB (dictionary key : bbox)
        process : Iou 계산
        output : IoU score (float)
        """
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea)
    @staticmethod
    def compute_residuals(pred_box, gt_box):
        """
        Residual(abs) 계산 (x1 - x1_hat, y1 - y1_hat, x2 - x2_hat, y2 - y2_hat)
        input : 예측한 box, Gt Box
        process : Residual 계산
        output : 비순응 점수
        """
        return [
            abs(pred_box[0] - gt_box[0]),
            abs(pred_box[1] - gt_box[1]),
            abs(pred_box[2] - gt_box[2]),
            abs(pred_box[3] - gt_box[3]),
        ]
    
class CalibrationMetrics:
    @staticmethod
    def compute_classwise_quantiles(aggregated_results, alpha=0.1):
        """
        Class 별로 따로 quantile을 계산함
        """
        quantile_results = {}
        for cls, values in aggregated_results.items():
            values = np.array(values)
            if len(values) == 0:
                continue
            quantiles = np.quantile(values, q=1-alpha, axis=0, method='higher')
            quantile_results[cls] = {
                f"{int((1 - alpha) * 100)}%": quantiles.tolist()
            }
        return quantile_results

    @staticmethod
    def compute_global_quantile(aggregated_results, alpha=0.1):
        """
        전체 객체에 대한 quantile을 추출함.
        """
        all_scores = [v[0] for values in aggregated_results.values() for v in values]
        return float(np.quantile(all_scores, 1 - alpha, method='higher')) if all_scores else None

    @staticmethod
    def save_results(classwise_qhat, global_qhat, path):
        """
        결과를 저장
        """
        with open(path, "w") as f:
            json.dump({
                "classwise_qhat": {str(k): v for k, v in classwise_qhat.items()},
                "global_qhat": global_qhat
            }, f, indent=2)

    @staticmethod
    def load_results(path):
        """
        결과를 불러옴
        """
        with open(path, "r") as f:
            return json.load(f)

class CalibrationRunner:
    """
    CalibrationRunner는 데이터셋을 로드하고, 예측기를 사용하여 각 이미지를 평가하는 클래스입니다.
    """
    def __init__(self, dataset_dicts, predictor):
        self.dataset_dicts = dataset_dicts
        self.predictor = predictor

    def evaluate_image(self, data):
        """
        이미지 평가: 이미지를 불러와서 예측을 한 다음 ioU가 0.2 이상인 박스에 대해 잔차를 계산합니다.
        """
        img = cv2.imread(data["file_name"])
        outputs = self.predictor(img)

        pred_boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
        pred_scores = outputs["instances"].scores.cpu().numpy()
        pred_classes = outputs["instances"].pred_classes.cpu().numpy()
        gt_boxes = [BoxMode.convert(ann["bbox"], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS) for ann in data["annotations"]]

        matched = set()
        results = {}

        for i, gt_box in enumerate(gt_boxes):
            best_idx, best_iou = -1, 0
            for j, pred_box in enumerate(pred_boxes):
                if j in matched:
                    continue
                iou = BoxCalculations.compute_iou(gt_box, pred_box)
                if iou >= 0.2 and iou > best_iou:
                    best_iou = iou
                    best_idx = j
            if best_idx != -1:
                matched.add(best_idx)
                score = pred_scores[best_idx]
                cls = pred_classes[best_idx]
                nonconf = 1 - score
                residuals = BoxCalculations.compute_residuals(pred_boxes[best_idx], gt_box)
                results.setdefault(cls, []).append((nonconf, *residuals))

        return results

    def run(self):
        """
        dataset_dicts에 있는 아이템들에 대해서, evaluate_image를 호출하여 결과를 집계합니다.
        """
        agg = {}
        for idx, data in enumerate(self.dataset_dicts):
            if idx % 500 == 0:
                print(f"Processing {idx}: {data['file_name']}")
            result = self.evaluate_image(data)
            for cls, values in result.items():
                agg.setdefault(cls, []).extend(values)
        return agg


def generate_calibration(image_root, json_file, predictor, alpha=0.1, output_file=None, save_agg=False):
    """
    캘리브레이션을 수행하고 결과를 저장하는 통합 함수
    
    Args:
        image_root (str): 이미지 루트 디렉토리 경로
        json_file (str): COCO 어노테이션 JSON 파일 경로
        predictor: 예측기 객체 (ConformalPredictor)
        alpha (float): 오류율 (default: 0.1)
        output_file (str): 캘리브레이션 결과 저장 파일명 (None인 경우 자동 생성)
        save_agg (bool): 집계 결과 저장 여부 (default: False)
        
    Returns:
        tuple: (classwise_qhat, global_qhat) - 생성된 캘리브레이션 결과
    """
    import time
    start_time = time.time()
    
    # 출력 파일명 자동 생성
    if output_file is None:
        confidence = int((1 - alpha) * 100)
        output_file = f"calib_q{confidence}.json"
    
    print(f"Loading dataset from {image_root}")
    loader = DatasetLoader(image_root, json_file)
    dataset = loader.load_dataset()
    
    print(f"Running calibration with alpha={alpha} (confidence={int((1-alpha)*100)}%)")
    runner = CalibrationRunner(dataset, predictor)
    results = runner.run()
    
    classwise_qhat = CalibrationMetrics.compute_classwise_quantiles(results, alpha=alpha)
    global_qhat = CalibrationMetrics.compute_global_quantile(results, alpha=alpha)
    
    print(f"Saving calibration results to {output_file}")
    CalibrationMetrics.save_results(classwise_qhat, global_qhat, output_file)
    
    # 집계 결과 저장 (옵션)
    if save_agg:
        # 출력 파일과 동일한 이름 패턴 사용
        agg_file = output_file.replace(".json", "_agg.json")
        if output_file == agg_file:  # 혹시 패턴이 일치하지 않으면
            agg_file = f"agg_result_{int((1-alpha)*100)}.json"
            
        print(f"Saving aggregated results to {agg_file}")
        with open(agg_file, "w") as f:
            serializable_results = {}
            for cls_id, values in results.items():
                serializable_results[str(cls_id)] = [list(v) for v in values]
            json.dump(serializable_results, f, indent=2)
    
    elapsed = time.time() - start_time
    print(f"Calibration completed in {elapsed:.2f} seconds")
    
    return classwise_qhat, global_qhat


if __name__ == "__main__":
    # test code
    image_root = "/home/datasets/coco/val2017"
    json_file = "/home/datasets/coco/annotations/instances_val2017.json"

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6
    cfg.MODEL.DEVICE = 'cuda'

    predictor = cppredictor.ConformalPredictor(cfg)

    loader = DatasetLoader(image_root, json_file)
    dataset = loader.load_dataset()

    runner = CalibrationRunner(dataset, predictor)
    resuls = runner.run()

    alpha = 0.1

    classwise_qhat = CalibrationMetrics.compute_classwise_quantiles(resuls, alpha=alpha)
    global_qhat = CalibrationMetrics.compute_global_quantile(resuls, alpha=alpha)
    CalibrationMetrics.save_results(classwise_qhat, global_qhat, "calib.json")
    
    # optional : save calibration dataset
    with open("agg_result.json", "w") as f:
        serializable_results = {}
        for cls_id, values in resuls.items():
            serializable_results[str(cls_id)] = [list(v) for v in values]
        json.dump(serializable_results, f, indent=2)    




    """
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
    """

