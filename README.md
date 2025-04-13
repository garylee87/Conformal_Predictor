# Conformal Object Detection

이 프로젝트는 [Facebook AI Research의 Detectron2](https://github.com/facebookresearch/detectron2)를 기반으로 한 Conformal Prediction을 객체 탐지에 적용한 구현체입니다.

## 프로젝트 소개

객체 탐지 모델(특히 Faster R-CNN)에 대한 불확실성 추정을 위해 Conformal Prediction 방법론을 적용했습니다. 이를 통해:

1. **적응형 예측 세트(Adaptive Prediction Set)**: 신뢰도 수준에 따라 객체의 클래스 예측을 여러 후보로 확장
2. **바운딩 박스 불확실성 추정**: 객체 위치의 불확실성을 정량화하여 예측 영역 확장
3. **확률적 보증**: 사용자가 지정한 신뢰도 수준(예: 90%)을 통계적으로 보장

<div align="center">
  <img src="https://user-images.githubusercontent.com/1381301/66535560-d3422200-eace-11e9-9123-5535d469db19.png"/>
  <p><i>Detectron2 기반 객체 탐지의 예시 이미지</i></p>
</div>

## 주요 기능

- **캘리브레이션 평가**: 유효성 검증 데이터셋을 사용한 모델 캘리브레이션
- **적응형 예측 세트(APS)**: 클래스 확률 기반 예측 세트 생성
- **바운딩 박스 확장**: 캘리브레이션 기반 불확실성 영역 계산
- **통계적 보증**: 미지의 테스트 데이터에 대한 오류율 제한