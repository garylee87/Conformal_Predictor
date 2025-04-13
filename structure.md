detectron2/
├── utils/
│   ├── __init__.py
│   ├── hooks.py         # 다양한 Hook 클래스 (LogitsHook, BoxesHook)
│   ├── label_utils.py   # 레이블 관련 유틸리티 (coco_label_from_index)
│   └── box_utils.py     # 바운딩 박스 관련 유틸리티
├── predictors/
│   ├── __init__.py
│   ├── base_predictor.py      # 기본 예측자 클래스
│   └── conformal_predictor.py # ConformalPredictor 클래스
└── evaluation/
    ├── __init__.py
    └── calibration.py   # 캘리브레이션 관련 함수와 클래스