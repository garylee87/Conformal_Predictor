import torch

def coco_label_from_index(indices):
    """
    주어진 1차원 텐서, 리스트 또는 단일 정수를 COCO 라벨 텍스트로 변환

    Args:
        indices: int, list 또는 torch.Tensor - 변환할 클래스 인덱스
    
    Returns:
        str 또는 str 리스트 - 변환된 COCO 라벨
        
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

    if isinstance(indices, torch.Tensor):
        indices = indices.tolist()

    if isinstance(indices, int):
        return coco_labels[indices] if 0 <= indices < len(coco_labels) else "Invalid index"

    return [coco_labels[i] if 0 <= i < len(coco_labels) else "Invalid index" for i in indices]

if __name__ == "__main__":
    IdxVec = torch.tensor([0, 1, 2, 10, 79, 80])
    idx = int(1)
    print(coco_label_from_index(IdxVec))
    print(coco_label_from_index(idx))