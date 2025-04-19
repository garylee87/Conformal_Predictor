import cv2
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

def visualize_conformal_predictions(image, outputs, cfg, scale=1.0, alpha=0.2):
    """
    이미지와 모델 출력 결과에서 Conformal Region을 시각화하는 함수.
    
    Args:
        image (np.ndarray): 원본 이미지 (BGR 포맷)
        outputs (dict): 모델 예측 결과 (instances 포함)
        cfg: detectron2 설정 객체
        scale (float): visualizer scaling factor (default: 1.0)
        alpha (float): 투명도 (0=투명, 1=불투명)
    
    Returns:
        output_img (np.ndarray): Conformal Region Box가 그려진 이미지
    """
    v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=scale)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    output_img = out.get_image()[:, :, ::-1].copy()
    conformal_regions = outputs["instances"].conformal_region
    
    for region in conformal_regions:
        # 좌표 추출
        x1_min = int(region["x1_min"])
        y1_min = int(region["y1_min"])
        x2_max = int(region["x2_max"])
        y2_max = int(region["y2_max"])
        
        x1_max = int(region["x1_max"])
        y1_max = int(region["y1_max"])
        x2_min = int(region["x2_min"])
        y2_min = int(region["y2_min"])
        
        # 1. 투명한 영역 생성을 위한 오버레이 이미지 생성
        overlay = output_img.copy()
        
        # 2. 4개의 영역을 채우기 (inner box와 outer box 사이의 영역)
        # 왼쪽 영역
        cv2.rectangle(overlay, (x1_min, y1_min), (x1_max, y2_max), (0, 255, 0), -1)
        # 오른쪽 영역
        cv2.rectangle(overlay, (x2_min, y1_min), (x2_max, y2_max), (0, 255, 0), -1)
        # 위쪽 영역
        cv2.rectangle(overlay, (x1_max, y1_min), (x2_min, y1_max), (0, 255, 0), -1)
        # 아래쪽 영역
        cv2.rectangle(overlay, (x1_max, y2_min), (x2_min, y2_max), (0, 255, 0), -1)
        
        # 3. 알파 블렌딩 적용
        output_img = cv2.addWeighted(overlay, alpha, output_img, 1 - alpha, 0)
        
        # 4. 테두리 그리기 (투명 영역 위에)
        # outer box
        cv2.rectangle(output_img, (x1_min, y1_min), (x2_max, y2_max), (0, 255, 0), 1)
        # inner box
        cv2.rectangle(output_img, (x1_max, y1_max), (x2_min, y2_min), (0, 255, 0), 1)
    
    return output_img

# 추가적인 시각화 기능을 여기에 구현할 수 있습니다.
def visualize_with_ground_truth(image, outputs, cfg, annotations, scale=1.0):
    """
    Conformal Prediction 결과와 Ground Truth를 함께 시각화하는 함수.
    
    Args:
        image (np.ndarray): 원본 이미지 (BGR 포맷)
        outputs (dict): 모델 예측 결과 (instances 포함)
        cfg: detectron2 설정 객체
        annotations (list): COCO 형식의 ground truth 어노테이션
        scale (float): visualizer scaling factor (default: 1.0)
    
    Returns:
        output_img (np.ndarray): Conformal Region과 Ground Truth가 그려진 이미지
    """
    # 먼저 conformal predictions 그리기
    output_img = visualize_conformal_predictions(image, outputs, cfg, scale)
    
    # Ground truth 바운딩 박스 추가 (점선 빨간색)
    for ann in annotations:
        if 'bbox' in ann:
            bbox = ann['bbox']  # XYWH 형식
            x, y, w, h = [int(v) for v in bbox]
            # XYWH -> XYXY 변환
            x1, y1, x2, y2 = x, y, x+w, y+h
            # 점선 효과를 위한 함수 호출
            draw_dashed_rectangle(output_img, (x1, y1), (x2, y2), (0, 0, 255), 1, dash_length=8)
            
    return output_img

def draw_dashed_rectangle(img, pt1, pt2, color, thickness=1, dash_length=10):
    """
    점선 사각형을 그리는 함수.
    
    Args:
        img: 이미지
        pt1 (tuple): 좌상단 좌표 (x1, y1)
        pt2 (tuple): 우하단 좌표 (x2, y2)
        color (tuple): BGR 색상
        thickness (int): 선 두께
        dash_length (int): 대시 길이
    """
    x1, y1 = pt1
    x2, y2 = pt2
    
    # 사각형의 4개 변을 개별적으로 그림
    # 위쪽 변
    for x in range(x1, x2, dash_length*2):
        end = min(x + dash_length, x2)
        cv2.line(img, (x, y1), (end, y1), color, thickness)
    
    # 오른쪽 변
    for y in range(y1, y2, dash_length*2):
        end = min(y + dash_length, y2)
        cv2.line(img, (x2, y), (x2, end), color, thickness)
    
    # 아래쪽 변
    for x in range(x2, x1, -dash_length*2):
        start = max(x - dash_length, x1)
        cv2.line(img, (start, y2), (x, y2), color, thickness)
    
    # 왼쪽 변
    for y in range(y2, y1, -dash_length*2):
        start = max(y - dash_length, y1)
        cv2.line(img, (x1, start), (x1, y), color, thickness)