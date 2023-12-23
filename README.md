# GradCam

# 01. Grad-CAM


# 02. Grad-CAM을 적용할 층 선택
## why? Grad-CAM을 이용해 중요 영역을 시각적 표현 > 랜드마크 가중치로 세부적인 부분을 강조 
1. ResNet의 마지막 합성곱 층
> 가장 복잡한 고차원 특징을 포착
> 
> 어떤 고차원 특징에 주목하는지 이해 가능


2. AdaptiveAvgPool2d 층 이전
> 이미지의 공간적 정보가 여전히 보존되어 있는 상태
> 
> 어떤 공간적 특징을 중시하는지를 파악


```
!pip install grad-cam
```

```
from pytorch_grad_cam import GradCAM
```

```
model = model

# 실험 1. ResNet의 마지막 합성곱 층
target_layers = [model.layer4[-1].conv2]

# CAM 생성하기 
cam = GradCAM(model=model, target_layers=target_layers)
```
