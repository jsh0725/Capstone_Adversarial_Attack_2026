# Seg_Attack 실험 요약

이 문서는 현재 VHRV 실험 설정, 생성자 가중치, 공격 이미지 생성 방식, 출력물 구조, 최신 정량 결과를 정리한 문서입니다. GitHub README의 실험 설명 섹션으로 그대로 사용할 수 있도록 구성했습니다.

## 개요

`Seg_Attack`은 VHRV 선박 탐지 데이터셋에서 객체 탐지 모델을 대상으로 adversarial perturbation 기반 공격을 평가합니다. 현재 최우선 실행 스크립트는 `evaluate_val_4modes.py`이며, 이 스크립트는 네 가지 모드의 이미지를 생성하고 YOLO detector 성능 및 이미지 품질 지표를 함께 평가합니다.

| Mode | 설명 |
|---|---|
| `original` | 공격 없는 VHRV validation 이미지 baseline |
| `fabricate` | `G_f` perturbation을 배경 영역에 적용하여 오탐을 유도하는 공격 |
| `vanish` | `G_v` perturbation을 객체 영역에 적용하여 미탐을 유도하는 공격 |
| `janusnet` | 제안 방법론 이름. `G_v`, `G_f`, FastSAM mask를 이용해 두 공격을 공간적으로 결합 |

중요: 이 프로젝트에서 `JanusNET`은 외부에서 다운로드 가능한 별도 모델이나 공개 아키텍처를 의미하지 않습니다. 현재 코드 기준으로는 학습된 두 생성자 `G_v`, `G_f`의 출력을 FastSAM mask로 결합하는 제안 방법론의 이름입니다.

## 주요 파일

| 파일 | 역할 |
|---|---|
| `evaluate_val_4modes.py` | 메인 평가 스크립트. 4개 모드 이미지 생성, detector 평가, LPIPS/SSIM/PSNR/speed 계산, 예시 이미지 저장 |
| `train.py` | `G_v`, `G_f` 생성자 학습 및 epoch별 검증 지표 저장 |
| `model.py` | `G_v`, `G_f`에 공통으로 사용되는 `MobileUNetGenerator` 정의 |
| `dataset.py` | YOLO 형식 데이터셋 로더. VHRV의 `images/val`, `labels/val` 구조 지원 |
| `config.py` | 이미지 크기, 배치, attack budget, 정규화 계수 등 실험 기본값 정의 |
| `utils.py` | YOLO raw logits, GT mask, anchor weight 관련 유틸리티 |
| `experiment_results.csv` | 생성자 학습 및 검증 로그 |

## 데이터셋 및 가중치

현재 실험은 `vhrv` 폴더 안의 VHRV 데이터셋과 VHRV로 학습한 생성자 가중치를 사용합니다.

```text
vhrv/
  vhrv.yaml
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```

`evaluate_val_4modes.py`의 기본 경로는 다음과 같습니다.

| 항목 | 경로 |
|---|---|
| Dataset yaml | `vhrv/vhrv.yaml` |
| Detector weights | `vhrv_yolo11x.pt` |
| Vanish generator | `results/vhrv_vresults/G_v_best.pth` |
| Fabricate generator | `results/vhrv_fresults/G_f_best.pth` |
| FastSAM weights | `FastSAM-s.pt` |

## 공격 이미지 생성 방식

모든 모드는 먼저 원본 validation 이미지를 `640x640` 정사각형으로 letterbox resize합니다. YOLO label도 동일한 letterbox 좌표계로 변환하여 저장합니다.

### Original

```python
adv = img
```

원본 이미지를 letterbox한 뒤 그대로 저장하며, detector baseline 평가에 사용합니다.

### Fabricate

```python
nf = G_f(img)
nf_c = clamp(nf, -EPSILON_F, EPSILON_F)
adv = clamp(img + nf_c * (1 - gt_mask), 0, 1)
```

`G_f`는 오탐 유도용 perturbation을 생성합니다. 이 perturbation은 GT bbox mask의 반대 영역, 즉 배경 또는 비객체 영역에 적용됩니다.

### Vanish

```python
nv = G_v(img)
nv_c = clamp(nv, -EPSILON_V, EPSILON_V)
adv = clamp(img + nv_c * gt_mask, 0, 1)
```

`G_v`는 미탐 유도용 perturbation을 생성합니다. 이 perturbation은 GT bbox mask로 정의된 객체 영역에 적용됩니다.

### JanusNET

```python
nv = G_v(img)
nf = G_f(img)

nv_c = clamp(nv, -EPSILON_V, EPSILON_V)
nf_c = clamp(nf, -EPSILON_F, EPSILON_F)

sam_obj = FastSAM_union_mask(img)
adv = clamp(img + nv_c * sam_obj + nf_c * (1 - sam_obj), 0, 1)
```

`JanusNET`은 별도 생성자를 새로 학습하지 않습니다. 이미 학습된 `G_v`, `G_f`의 출력을 FastSAM union mask 기준으로 결합합니다. FastSAM이 객체 후보로 본 영역에는 `G_v` perturbation을 적용하고, 그 외 영역에는 `G_f` perturbation을 적용합니다.

## 실행 방법

기본 실행:

```powershell
python evaluate_val_4modes.py
```

명시 실행:

```powershell
python evaluate_val_4modes.py `
  --data .\vhrv\vhrv.yaml `
  --split val `
  --weights .\vhrv_yolo11x.pt `
  --gv .\results\vhrv_vresults\G_v_best.pth `
  --gf .\results\vhrv_fresults\G_f_best.pth `
  --fastsam-weights .\FastSAM-s.pt `
  --variants original fabricate vanish janusnet `
  --imgsz 640 `
  --batch 1 `
  --outdir .\eval_outputs
```

## 출력물 구조

`evaluate_val_4modes.py` 실행 후 기본적으로 `eval_outputs/` 아래에 결과가 저장됩니다.

```text
eval_outputs/
  val_4mode_metrics.csv
  val_original.yaml
  val_fabricate.yaml
  val_vanish.yaml
  val_janusnet.yaml
  generated_val_sets/
    original/
      images/
      labels/
    fabricate/
      images/
      labels/
    vanish/
      images/
      labels/
    janusnet/
      images/
      labels/
  examples/
    fabricate/
    vanish/
    janusnet/
```

`val_4mode_metrics.csv`는 최종 정량 결과 파일입니다. `examples/`에는 원본 예측, 공격 이미지, perturbation 시각화, 공격 후 예측, detection 변화 통계가 포함된 예시 패널이 저장됩니다.

## 생성자 학습 지표

현재 평가에 사용한 생성자 가중치는 `experiment_results.csv`의 `VHRV_V`, `VHRV_F` 학습 결과입니다.

| 생성자 | Epoch | Loss | Attack 지표 | PSNR | SSIM | LPIPS | Linf |
|---|---:|---:|---:|---:|---:|---:|---:|
| `G_v` Vanish | 20 | `lv=0.1345` | `v_missed=1.4375`, `v_new=0.0625` | 41.1908 | 0.9923 | 0.0160 | 0.062745 |
| `G_f` Fabricate | 20 | `lf=12.0799` | `f_missed=0.5000`, `f_new=295.7500` | 37.4070 | 0.9211 | 0.1346 | 0.015686 |

`G_v`는 높은 시각적 유사성을 유지하면서 미탐을 유도하는 데 초점을 둔 생성자입니다. `G_f`는 배경 영역에서 다수의 신규 탐지를 유도하는 오탐 공격 생성자이며, `G_v`보다 시각적 변화가 상대적으로 크게 나타납니다.

## Detector 평가 결과

최신 VHRV validation 결과는 `eval_outputs/val_4mode_metrics.csv` 기준입니다.

| Mode | mAP50 | Precision | Recall | F1 | LPIPS | SSIM | PSNR | Speed (s/img) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original | 0.9551 | 0.9772 | 0.8998 | 0.9369 | 0.0000 | 1.0000 | inf | 0.0200 |
| Fabricate | 0.7223 | 0.9867 | 0.5613 | 0.7155 | 0.1345 | 0.9954 | 37.38 | 0.0254 |
| Vanish | 0.8463 | 0.9455 | 0.7065 | 0.8087 | 0.0153 | 0.9974 | 41.47 | 0.0246 |
| JanusNET | 0.3797 | 0.8583 | 0.2892 | 0.4326 | 0.2962 | 0.9760 | 30.83 | 0.0457 |

Original 대비 JanusNET은 `mAP50`을 `0.9551`에서 `0.3797`로, `Recall`을 `0.8998`에서 `0.2892`로, `F1`을 `0.9369`에서 `0.4326`으로 감소시켰습니다. 이는 미탐 공격과 오탐 공격을 공간적으로 결합한 통합 공격이 단일 공격보다 더 큰 detector 성능 저하를 유도함을 보여줍니다.

다만 JanusNET은 `LPIPS=0.2962`, `PSNR=30.83`으로 네 모드 중 시각적 왜곡이 가장 크게 나타났습니다. 향후에는 perturbation budget 감소, L2/TV/color regularization 강화, FastSAM mask 정제 등을 통해 공격 성능과 시각적 은밀성 사이의 trade-off를 조정할 수 있습니다.

## 지표 설명

| 지표 | 의미 | 해석 방향 |
|---|---|---|
| `mAP50` | IoU 0.50 기준 평균 정밀도 | 낮을수록 공격 효과가 큼 |
| `Precision` | 예측한 객체 중 정답 비율 | 낮아지면 오탐 증가 가능성 |
| `Recall` | 실제 객체 중 탐지 성공 비율 | 낮아지면 미탐 증가 |
| `F1` | Precision과 Recall의 조화 평균 | 낮을수록 전체 탐지 성능 저하 |
| `PSNR` | 원본 대비 픽셀 단위 왜곡 정도 | 높을수록 원본과 유사 |
| `SSIM` | 구조적 유사도 | 높을수록 구조적으로 유사 |
| `LPIPS` | 딥러닝 feature 기반 perceptual distance | 낮을수록 사람 눈에 유사 |
| `Linf` | 최대 perturbation 크기 | 낮을수록 최대 픽셀 변화가 작음 |

## 학습 및 재학습 메모

JanusNET을 실행하기 위해 `G_v`, `G_f`를 다시 학습할 필요는 없습니다. JanusNET은 학습된 두 생성자의 출력을 inference 단계에서 결합하는 방식입니다.

재학습은 공격 효과와 시각적 왜곡 사이의 균형을 다시 맞추고 싶을 때 필요합니다.

| 조정 항목 | 예상 효과 |
|---|---|
| `EPSILON_V`, `EPSILON_F` 감소 | 시각적 왜곡 감소, 공격 성능 약화 가능 |
| `LAMBDA_L2` 증가 | 전체 perturbation 크기 억제 |
| `GAMMA_TV` 증가 | 거친 노이즈 패턴 완화 |
| `LAMBDA_COL` 증가 | 색상 변화 및 컬러 artifact 억제 |
| FastSAM mask filtering/refinement | JanusNET에서 과도하게 넓은 mask 적용을 줄여 왜곡 완화 가능 |

## 향후 계획

향후 연구에서는 정지 이미지 기반 공격을 영상 기반 탐지 환경으로 확장할 예정입니다. 목표는 제안한 공격이 단일 이미지뿐 아니라 연속 프레임에서도 detector 성능을 일관되게 저하시킬 수 있음을 검증하는 것입니다. 이를 위해 프레임 단위 탐지 성능 변화, perturbation의 시간적 일관성, 영상 내 탐지 또는 추적 안정성 변화를 함께 분석할 계획입니다.