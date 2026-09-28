# AI4I Predictive Maintenance Analysis

AI4I 2020 Predictive Maintenance Dataset을 활용하여 설비의 운전 조건을 분석하고,  
TWF, HDF, PWF, OSF, RNF의 5가지 고장 유형을 판별하는 프로젝트입니다.

## 1. 폴더 구조

```text
ai4i-predictive-maintenance-analysis/
│
├── data/
│   └── ai4i2020.csv
│
├── main.py
├── data_utils.py
├── feature_utils.py
├── model_utils.py
├── threshold_utils.py
├── visualization.py
├── korean_font.py
└── README.md
```

각 파일의 역할은 다음과 같습니다.

- `main.py` : 전체 분석 과정 실행
- `data_utils.py` : 데이터 불러오기, X/y 구성, Train/Test 분할
- `feature_utils.py` : 파생변수 생성
- `model_utils.py` : Z-score Baseline 및 Random Forest 모델 학습·평가
- `threshold_utils.py` : Threshold 및 FP/FN 비용 분석
- `visualization.py` : EDA 및 모델 결과 시각화
- `korean_font.py` : 그래프 한글 폰트 설정
- `data/ai4i2020.csv` : 분석에 사용하는 원본 데이터

---

## 2. 데이터 출처

본 프로젝트는 UCI Machine Learning Repository의  
**AI4I 2020 Predictive Maintenance Dataset**을 사용합니다.

데이터 출처:

https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

주요 고장 유형은 다음과 같습니다.

- `TWF` : Tool Wear Failure
- `HDF` : Heat Dissipation Failure
- `PWF` : Power Failure
- `OSF` : Overstrain Failure
- `RNF` : Random Failure

---

## 3. 데이터 다운로드

1. 위 UCI 데이터 페이지에 접속합니다.
2. 데이터셋을 다운로드합니다.
3. 다운로드한 `ai4i2020.csv` 파일을 프로젝트의 `data` 폴더에 넣습니다.

최종적으로 다음 경로에 파일이 존재해야 합니다.

```text
ai4i-predictive-maintenance-analysis/data/ai4i2020.csv
```

`data` 폴더가 없다면 프로젝트 루트에 직접 생성합니다.

---

## 4. 실행 방법

### 4-1. 프로젝트 폴더로 이동

터미널에서 프로젝트가 저장된 위치로 이동합니다.

```bash
cd ai4i-predictive-maintenance-analysis
```

현재 위치에 `main.py`와 `data` 폴더가 함께 있는지 확인합니다.

```bash
ls
```

### 4-2. 필요한 패키지 설치

```bash
pip install pandas numpy matplotlib scikit-learn
```

### 4-3. 프로그램 실행

```bash
python main.py
```

Windows 환경에서 `python` 명령어가 동작하지 않는 경우 다음과 같이 실행할 수 있습니다.

```bash
py main.py
```

---

## 5. 데이터 경로 주의사항

`main.py`에서는 다음과 같이 데이터를 불러옵니다.

```python
df = load_data("data/ai4i2020.csv")
```

따라서 `ai4i2020.csv`가 반드시 `data` 폴더 안에 있어야 합니다.

다음과 같이 파일이 프로젝트 루트에 직접 위치하면 정상적으로 실행되지 않습니다.

```text
ai4i-predictive-maintenance-analysis/
├── ai4i2020.csv    # 잘못된 위치
└── main.py
```

반드시 다음 구조를 사용합니다.

```text
ai4i-predictive-maintenance-analysis/
├── data/
│   └── ai4i2020.csv
└── main.py
```

---

## 6. 팀원 및 역할

| 팀원 | 담당 역할 | 주요 내용 |
|---|---|---|
| 유다빈 | 프로젝트 총괄 및 분석 통합 | EDA 방향 설정, Feature Engineering, Threshold 및 비용 분석, 전체 분석 결과 통합 |
| 권률 | 고장 원인 해석 및 문서화 | 고장 유형별 특성 및 원인 해석, 분석 결과 정리 및 문서화 |
| 김수혁 | 모델링 | 고장 유형별 분류 모델 구축 및 모델 성능 비교 |
| 김재욱 | 데이터 분석 | 데이터 구조 및 품질 확인, 설비 Type별 고장 건수 및 고장률 분석 |
| 유인서 | 모델 평가 | 센서 및 특성 분석 | 센서 간 관계 분석, 시계열 특성 및 파생변수 분석 |

### 협업 방식

각 담당자는 개별 분석 결과와 코드를 GitHub Branch에서 관리하고,  
분석 결과를 공유하여 최종 분석 과정에 통합합니다.

프로젝트의 전체 분석 흐름은 다음과 같습니다.

`데이터 이해 및 전처리 → EDA → Feature Engineering → 모델링 → 모델 평가 → Threshold 및 비용 분석 → 결과 해석`