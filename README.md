# Conformal Object Detection

이 프로젝트는 [Facebook AI Research의 Detectron2](https://github.com/facebookresearch/detectron2)를 기반으로 한 Conformal Prediction을 객체 탐지에 적용한 구현체입니다.

## 프로젝트 소개

객체 탐지 모델(특히 Faster R-CNN)에 대한 불확실성 추정을 위해 Conformal Prediction 방법론을 적용했습니다. 이를 통해:

1. **적응형 예측 세트(Adaptive Prediction Set)**: 신뢰도 수준에 따라 객체의 클래스 예측을 여러 후보로 확장
2. **바운딩 박스 불확실성 추정**: 객체 위치의 불확실성을 정량화하여 예측 영역 확장
3. **확률적 보증**: 사용자가 지정한 신뢰도 수준(예: 90%)을 통계적으로 보장

<img src = "/home/RCNN/detectron2/output_conformal.jpg">

## 주요 기능

- **캘리브레이션 평가**: 유효성 검증 데이터셋을 사용한 모델 캘리브레이션
- **적응형 예측 세트(APS)**: 클래스 확률 기반 예측 세트 생성
- **바운딩 박스 확장**: 캘리브레이션 기반 불확실성 영역 계산
- **통계적 보증**: 미지의 테스트 데이터에 대한 오류율 제한

## 관련 연구 및 참고 자료
이 프로젝트는 다음 연구를 기반으로 합니다:
 - [Adaptive Bounding Box Uncertainties via Two-Step Conformal Prediction
](https://eccv2024.ecva.net/media/eccv-2024/Slides/139.pdf)
 - [A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification
](https://arxiv.org/abs/2107.07511)
 - [Object Detection with Probabilistic Guarantees: A Conformal Prediction Approach](https://link.springer.com/chapter/10.1007/978-3-031-14862-0_23)



## 라이선스
이 프로젝트는 원본 Detectron2와 동일한 Apache 2.0 라이선스를 따릅니다.

## 원본 Detectron2
이 프로젝트는 Facebook AI Research의 Detectron2를 기반으로 합니다.

