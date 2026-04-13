# Experiment Plan

이 문서는 확대 시 육안 식별 흔적을 줄이기 위한 하이퍼파라미터 실험표다.
기본 원칙은 한 번에 한 축만 바꾸고, 공격 성능과 은닉성 지표를 같이 보는 것이다.

## Current Read

현재 남아 있는 결과물 기준으로는 `vanish`가 학습 후반부에 거의 plateau에 들어간 것으로 보인다.
즉 `vanish`는 "더 오래 학습"보다는 "덜 보이도록 미세조정"이 우선이다.
반면 `fabricate`는 여전히 조정 여지가 더 크다.

## Baseline

현재 기본값은 아래와 같다.

```python
IMG_SIZE = 640
BATCH_SIZE = 2
EPOCHS = 20
EARLY_STOP_PATIENCE = 4
VAL_MAX_BATCHES = 12
NUM_WORKERS = 2
PIN_MEMORY = 1
EPSILON_V = 16 / 255.0
EPSILON_F = 4 / 255.0
HINGE_CAP = 0.5
LAMBDA_L2 = 15.0
GAMMA_TV = 1.5
LAMBDA_COL = 3.0
VANISH_LAMBDA_L2 = 0.0
VANISH_GAMMA_TV = 0.0
VANISH_LAMBDA_COL = 0.0
```

## What To Watch

실험마다 아래 항목을 같이 기록한다.

- `Vanish`: `missed`, `new`, `PSNR`, `Linf`
- `Fabricate`: `new`, `missed`, `PSNR`, `Linf`
- `results/epXXX_*.png` 확대 확인

권장 해석 기준:

- `PSNR`가 올라가고 `Linf`가 내려가면 대체로 시각 흔적이 줄어든다.
- `Vanish`는 `missed`가 너무 크게 떨어지지 않는 범위에서 흔적 감소를 찾는다.
- `Fabricate`는 `new`가 유지되면서 배경 잡티와 색 번짐이 줄어드는 조합을 찾는다.

## Decision Rule

실험 채택 기준은 아래처럼 정한다.

### Vanish

- 목적: 추가 suppression 극대화보다 확대 시 식별성 감소
- 기준선: 저장된 과거 결과 기준 대략 `missed 37~39`, suppression `70% 전후`
- 채택 조건:
- `v_missed >= 34`
- `v_new <= 3`
- `psnr_v`는 baseline 이상
- `linf_v`는 baseline 이하
- 확대 이미지에서 컬러 링잉, 고주파 입자감, 경계 번짐이 감소

### Fabricate

- 목적: hallucination 유지 + 배경 노이즈 완화
- 채택 조건:
- `f_new`가 baseline 대비 `80%` 이상 유지
- `f_missed`는 baseline보다 증가하지 않는 방향
- `psnr_f` 상승
- `linf_f` 하락
- 확대 이미지에서 색 번짐과 배경 얼룩 감소

## RTX 4070 SUPER Preset

현재 장비가 `GeForce RTX 4070 SUPER 12GB`이므로 기본 실행 프리셋은 아래로 둔다.

```bash
IMG_SIZE=640
BATCH_SIZE=4
VAL_MAX_BATCHES=16
NUM_WORKERS=8
PIN_MEMORY=1
PREFETCH_FACTOR=2
PERSISTENT_WORKERS=1
```

문제가 생기면 아래 순서로 낮춘다.

1. `BATCH_SIZE=1`
2. `BATCH_SIZE=2`
3. `NUM_WORKERS=4`
4. `VAL_MAX_BATCHES=12`

## How To Apply

### Option 1. `config.py` 직접 수정

[`config.py`](/mnt/c/Users/user/Documents/Attack/config.py)에서 해당 값만 바꾼다.

### Option 2. 환경변수로 실행

기본값은 유지하고 실행할 때만 덮어쓴다.

```bash
EPSILON_V=0.0471 VANISH_LAMBDA_L2=5 VANISH_GAMMA_TV=1.0 VANISH_LAMBDA_COL=1.0 python3 train.py
```

실험 로그를 구분하려면 `EXPERIMENT_ID`를 함께 넣는다.

```bash
EXPERIMENT_ID=V5 EPSILON_V=0.0471 VANISH_LAMBDA_L2=5 VANISH_GAMMA_TV=1.0 python3 train.py
```

실행 결과는 기본적으로 [`experiment_results.csv`](/mnt/c/Users/user/Documents/Attack/experiment_results.csv) 에 epoch별로 누적 저장된다.

## Vanish Sweep

`vanish`는 이미 상당 부분 수렴한 것으로 보이므로, 이제는 후보 수를 줄여서 stealth 중심으로 재조정한다.

실행 시 모드는 `1) Vanish only`를 선택한다.

| ID | 목적 | EPSILON_V | VANISH_LAMBDA_L2 | VANISH_GAMMA_TV | VANISH_LAMBDA_COL | 기대 효과 |
|---|---|---:|---:|---:|---:|---|
| V0 | 기준선 | 16/255 | 0.0 | 0.0 | 0.0 | 현재 결과 재현 |
| V1 | epsilon만 낮춤 | 12/255 | 0.0 | 0.0 | 0.0 | 가장 빠른 흔적 감소 확인 |
| V2 | epsilon 더 낮춤 | 10/255 | 0.0 | 0.0 | 0.0 | suppression 유지 한계 확인 |
| V3 | L2만 추가 | 12/255 | 3.0 | 0.0 | 0.0 | 전체 노이즈 세기 억제 |
| V4 | TV까지 추가 | 12/255 | 5.0 | 1.0 | 0.0 | 거친 패턴 완화 |
| V5 | Color까지 추가 | 12/255 | 5.0 | 1.0 | 0.5 | 컬러 링잉 억제 |
| V6 | 보수형 균형안 | 10/255 | 5.0 | 1.0 | 1.0 | 성능과 은닉성 균형 후보 |
| V7 | 초보수형 | 8/255 | 8.0 | 2.0 | 1.0 | 육안 흔적 최소화 우선 |

### Vanish 추천 순서

후보를 적게 돌려도 방향성이 보이도록 아래 순서를 권장한다.

1. `V0 -> V1`
2. `V1`이 괜찮으면 `V4`
3. 시각 흔적이 여전히 거슬리면 `V5`
4. 성능 저하를 감수하고라도 자연스러움이 필요하면 `V6 -> V7`

### Vanish 빠른 추천

팀원이 말한 문제가 "확대하면 너무 보인다"에 가깝다면 아래 3개만 먼저 본다.

- `V1`
- `V4`
- `V6`

### Vanish 실행 예시

```bash
EXPERIMENT_ID=V4 EPSILON_V=0.0471 VANISH_LAMBDA_L2=5 VANISH_GAMMA_TV=1.0 BATCH_SIZE=2 IMG_SIZE=640 python3 train.py
```

## Fabricate Sweep

`fabricate`는 이미 stealth 항이 있어서 너무 많이 건드리기보다 보수적으로 조절한다.

실행 시 모드는 `2) Fabricate only`를 선택한다.

| ID | 목적 | EPSILON_F | HINGE_CAP | LAMBDA_L2 | GAMMA_TV | LAMBDA_COL | 기대 효과 |
|---|---|---:|---:|---:|---:|---:|---|
| F0 | 기준선 | 4/255 | 0.50 | 15.0 | 1.5 | 3.0 | 현재 결과 재현 |
| F1 | 세기만 감소 | 3/255 | 0.50 | 15.0 | 1.5 | 3.0 | 흔적 감소 첫 확인 |
| F2 | 세기 추가 감소 | 2/255 | 0.50 | 15.0 | 1.5 | 3.0 | fabricate 유지 한계 확인 |
| F3 | L2 강화 | 3/255 | 0.50 | 20.0 | 1.5 | 3.0 | 배경 교란 세기 감소 |
| F4 | TV 강화 | 3/255 | 0.50 | 20.0 | 2.5 | 3.0 | 배경 패턴 부드럽게 |
| F5 | Color 강화 | 3/255 | 0.50 | 20.0 | 2.5 | 4.0 | 컬러 노이즈 억제 |
| F6 | Hinge 완화 | 3/255 | 0.35 | 20.0 | 2.5 | 4.0 | 과도한 hallucination 압박 완화 |
| F7 | Hinge 더 완화 | 3/255 | 0.25 | 20.0 | 2.5 | 4.0 | 더 자연스럽지만 공격 약해질 수 있음 |
| F8 | 종합 보수형 | 2/255 | 0.35 | 30.0 | 4.0 | 6.0 | 은닉성 우선 후보 |
| F9 | 종합 타협형 | 3/255 | 0.35 | 25.0 | 3.0 | 5.0 | 성능과 품질 균형 후보 |

### Fabricate 추천 순서

1. `F0 -> F1 -> F2`
2. `F1`이 괜찮으면 `F3 -> F4 -> F5`
3. 마지막으로 `F6 -> F7 -> F9 -> F8`

### Fabricate 빠른 추천

먼저 아래 3개를 추천한다.

- `F1`
- `F5`
- `F9`

### Fabricate 실행 예시

```bash
EXPERIMENT_ID=F5 EPSILON_F=0.0118 LAMBDA_L2=20 GAMMA_TV=2.5 LAMBDA_COL=4 BATCH_SIZE=2 IMG_SIZE=640 python3 train.py
```

## Suggested Log Sheet

아래 형식으로 메모하면 비교가 쉬워진다.

```text
[V5]
EPSILON_V=12/255
VANISH_LAMBDA_L2=5.0
VANISH_GAMMA_TV=1.0
VANISH_LAMBDA_COL=0.5
v_missed=
v_new=
psnr_v=
linf_v=
accept/reject=
zoom visual note=
```

```text
[F5]
EPSILON_F=3/255
HINGE_CAP=0.50
LAMBDA_L2=20.0
GAMMA_TV=2.5
LAMBDA_COL=4.0
f_new=
f_missed=
psnr_f=
linf_f=
accept/reject=
zoom visual note=
```

## New Starting Set

지금 판단 기준을 반영하면 우선 아래 5개만 돌리면 된다.

- `V1`
- `V4`
- `V6`
- `F1`
- `F5`

## Summary

- `vanish`는 장기 학습보다 stealth 미세조정이 우선
- `fabricate`는 아직 sweep 폭을 조금 넓게 가져가도 됨
- CSV에는 epoch별 로그가 쌓이므로 최종 비교는 각 `EXPERIMENT_ID`의 마지막 3개 epoch를 함께 보는 것을 권장
