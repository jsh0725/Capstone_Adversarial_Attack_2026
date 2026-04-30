# Seg_Attack

VHRV 선박 탐지 데이터셋에서 YOLO detector를 대상으로 미탐/오탐 유도 perturbation을 생성하고, 개별 공격과 통합 공격의 성능 저하 효과를 비교한다.

현재 기준 실행 파일은 `evaluate_val_4modes.py`이다.

## 실험 구성

| 항목 | 값 |
|---|---|
| Dataset | `vhrv/vhrv.yaml` |
| Detector | `vhrv_yolo11x.pt` |
| Vanish generator | `results/vhrv_vresults/G_v_best.pth` |
| Fabricate generator | `results/vhrv_fresults/G_f_best.pth` |
| FastSAM | `FastSAM-s.pt` |
| Image size | `640` |
| Batch | `1` |

## 공격 모드

| Mode | 생성 방식 |
|---|---|
| `original` | 원본 validation 이미지 baseline |
| `vanish` | GT bbox 영역에 `G_v` perturbation 적용 |
| `fabricate` | GT bbox 외부 영역에 `G_f` perturbation 적용 |
| `janusnet` | FastSAM union mask 기준으로 `G_v`와 `G_f` perturbation 결합 |

`JanusNET`은 별도 외부 모델이 아니라, 본 실험에서 제안한 통합 공격 방식의 이름이다.

## 실행

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

기본 경로가 위 설정과 같다면 아래 명령만으로도 실행할 수 있다.

```powershell
python evaluate_val_4modes.py
```

## 생성자 학습 결과

| Generator | Epoch | Loss | Attack metric | PSNR | SSIM | LPIPS | Linf |
|---|---:|---:|---:|---:|---:|---:|---:|
| `G_v` | 20 | `lv=0.1345` | `v_missed=1.4375`, `v_new=0.0625` | 41.1908 | 0.9923 | 0.0160 | 0.062745 |
| `G_f` | 20 | `lf=12.0799` | `f_missed=0.5000`, `f_new=295.7500` | 37.4070 | 0.9211 | 0.1346 | 0.015686 |

`G_v`는 높은 이미지 유사도를 유지하면서 미탐을 유도했고, `G_f`는 배경 영역에서 신규 탐지를 크게 증가시켰다.

## Detector 평가 결과

VHRV validation set 기준 결과는 다음과 같다.

| Mode | mAP50 | Precision | Recall | F1 | LPIPS | SSIM | PSNR | Speed (s/img) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original | 0.9551 | 0.9772 | 0.8998 | 0.9369 | 0.0000 | 1.0000 | inf | 0.0200 |
| Fabricate | 0.7223 | 0.9867 | 0.5613 | 0.7155 | 0.1345 | 0.9954 | 37.38 | 0.0254 |
| Vanish | 0.8463 | 0.9455 | 0.7065 | 0.8087 | 0.0153 | 0.9974 | 41.47 | 0.0246 |
| JanusNET | 0.3797 | 0.8583 | 0.2892 | 0.4326 | 0.2962 | 0.9760 | 30.83 | 0.0457 |

JanusNET은 Original 대비 `mAP50`을 `0.9551`에서 `0.3797`로 낮췄고, `Recall`은 `0.8998`에서 `0.2892`로 감소시켰다. 단일 vanish/fabricate 공격보다 탐지 성능 저하가 크게 나타났으며, 동시에 LPIPS 증가와 PSNR 감소로 시각적 왜곡도 가장 크게 나타났다.

## 출력 파일

```text
eval_outputs/
  val_4mode_metrics.csv
  val_original.yaml
  val_fabricate.yaml
  val_vanish.yaml
  val_janusnet.yaml
  generated_val_sets/
  examples/
```

최종 정량 결과는 `eval_outputs/val_4mode_metrics.csv`에 저장된다. 예시 시각화 이미지는 `eval_outputs/examples/` 아래에 저장된다.

## 주요 지표

| 지표 | 의미 |
|---|---|
| `mAP50` | IoU 0.50 기준 detector 성능 |
| `Precision` | 예측한 객체 중 정답 비율 |
| `Recall` | 실제 객체 중 탐지 성공 비율 |
| `F1` | Precision과 Recall의 조화 평균 |
| `PSNR` | 픽셀 단위 원본 유사도, 높을수록 좋음 |
| `SSIM` | 구조적 유사도, 높을수록 좋음 |
| `LPIPS` | 지각적 차이, 낮을수록 좋음 |