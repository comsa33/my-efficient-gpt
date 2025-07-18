# 🧠 Efficient GPT: 뇌과학 기반 트랜스포머 최적화

[English Version](README.md)

nanoGPT에 뇌과학 기반 효율성 메커니즘을 적용하여 생성 품질을 유지하면서 최대 2배의 속도 향상을 달성한 연구 구현체입니다.

## 🌟 개요

이 프로젝트는 Andrej Karpathy의 nanoGPT를 확장하여 7가지 주요 뇌과학 기반 최적화를 통해 계산 효율성을 획기적으로 개선했습니다. 인간의 뇌가 단 20W의 전력으로 정보를 효율적으로 처리하는 방식을 모방하여, 다음과 같은 특징을 가진 트랜스포머 모델을 만들었습니다:

- 선택적 처리를 통한 **50-90% 연산량 감소**
- 더 적은 리소스로 **생성 품질 유지**
- 입력 복잡도에 따른 **동적 연산 조절**
- 재앙적 망각 없는 **점진적 학습 가능**

## 🚀 주요 기능

### 1. **예측 처리와 조기 종료 (Predictive Processing with Early Exit)** 🔮
- **영감**: 뇌의 예측 코딩 - 빠른 예측을 먼저 하고 필요할 때만 깊은 처리 수행
- **구현**: 각 레이어가 예측에 대한 확신도가 높을 때 조기 종료
- **효과**: "쉬운" 토큰에 대해 최대 50% 레이어 스킵
- **실생활 비유**: 익숙한 얼굴을 즉시 인식 vs. 낯선 사람을 자세히 관찰

### 2. **희소 활성화 패턴 (Sparse Activation Patterns)** 🎯
- **영감**: 뇌 뉴런의 1-5%만 동시에 활성화
- **구현**: MLP 레이어에서 동적 top-k 활성화 (10%만 활성)
- **효과**: MLP 연산량 90% 감소
- **실생활 비유**: 소음을 걸러내고 관련 세부사항에만 집중

### 3. **적응적 계산 시간 (Adaptive Computation Time)** ⏱️
- **영감**: 인간은 복잡한 문제에 더 많은 시간을 소비
- **구현**: 입력 난이도에 따른 가변적 처리 단계
- **효과**: 간단한 입력에서 2-5배 속도 향상
- **실생활 비유**: 빠른 암산 vs. 복잡한 문제 해결

### 4. **동적 라우팅 (Dynamic Routing)** 🛤️
- **영감**: 다른 작업을 위한 뇌의 특화된 영역
- **구현**: 콘텐츠 기반 전문가 네트워크 라우팅
- **효과**: 콘텐츠 유형별 특화된 처리
- **실생활 비유**: 단어는 언어 중추, 이미지는 시각 피질로

### 5. **로컬 어텐션 패턴 (Local Attention Patterns)** 🔍
- **영감**: 뇌의 국소 연결 패턴
- **구현**: 로컬 윈도우로 제한된 어텐션
- **효과**: O(n²) 대신 O(n*w) 복잡도
- **실생활 비유**: 전체 페이지가 아닌 주변 단어에 집중하여 읽기

### 6. **계층적 처리 (Hierarchical Processing)** 📊
- **영감**: 시각 피질 계층 구조 (V1→V2→V4)
- **구현**: 다양한 추상화 수준의 다중 스케일 표현
- **효과**: 지역적 및 전역적 패턴의 효율적 포착
- **실생활 비유**: 숲과 나무를 동시에 보기

### 7. **점진적 학습 (Incremental Learning)** 🔄
- **영감**: 시냅스 가소성과 기억 강화
- **구현**: EWC, 메모리 리플레이, 적응적 파라미터
- **효과**: 기존 지식을 잊지 않고 새로운 작업 학습
- **실생활 비유**: 기존 지식을 유지하면서 새로운 기술 습득

## 📊 성능 결과

셰익스피어 텍스트 생성 벤치마크 기준:

| 기능 | 속도 향상 | 메모리 감소 | 품질 영향 |
|------|-----------|-------------|-----------|
| 조기 종료 | 1.5-2배 | ~30% | < 1% 손실 |
| 희소 활성화 | 1.3-1.8배 | ~20% | < 0.5% 손실 |
| 로컬 어텐션 | 1.2-1.5배 | ~40% | < 2% 손실 |
| **전체 결합** | **1.91배** | **~50%** | **< 2% 손실** |

### 실제 성능 예시:
- **표준 GPT**: 66.5 토큰/초
- **Efficient GPT**: 127.2 토큰/초
- **속도 향상**: 1.91배 🚀

## 🛠️ 설치 방법

```bash
# 저장소 클론
git clone https://github.com/yourusername/my-efficient-gpt.git
cd my-efficient-gpt

# uv로 의존성 설치 (권장)
uv sync

# 또는 pip로 설치
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

## 🎮 빠른 시작

### 1. **인터랙티브 데모**
```bash
# 인터랙티브 데모 실행
uv run python simple_demo.py
```

### 2. **효율성 기능 테스트**
```bash
# 모든 기능 동작 확인
uv run python demo_efficiency.py
```

### 3. **모델 학습**
```bash
# 데이터 준비 (셰익스피어 예제)
cd data/shakespeare_char
uv run python prepare.py
cd ../..

# 효율성 기능으로 학습
uv run python train_efficient.py config/train_efficient_gpt.py
```

### 4. **성능 벤치마크**
```bash
# 표준 vs 효율적 모델 비교
uv run python test_efficiency.py
```

## 🔧 설정

`config/train_efficient_gpt.py`를 편집하여 기능 커스터마이즈:

```python
efficiency_config = {
    # 조기 종료
    'enable_early_exit': True,
    'exit_threshold': 0.95,      # 높을수록 더 적극적
    
    # 희소 활성화
    'enable_sparse_activation': True,
    'sparsity_ratio': 0.1,       # 상위 10% 유지
    
    # 적응적 계산
    'enable_adaptive_compute': True,
    'max_pondering_steps': 3,
    
    # 로컬 어텐션
    'enable_local_attention': True,
    'local_attention_window': 128,
    
    # 그 외...
}
```

## 📁 프로젝트 구조

```
my-efficient-gpt/
├── model.py                # 원본 GPT 구현
├── model_efficient.py      # 뇌과학 기반 기능이 추가된 Efficient GPT
├── efficient_modules.py    # 핵심 효율성 메커니즘
├── hierarchical_modules.py # 다중 스케일 처리
├── incremental_learning.py # 지속 학습 기능
├── train_efficient.py      # 학습 스크립트
├── demo_efficiency.py      # 기능 시연
├── simple_demo.py         # 빠른 테스트 스크립트
├── config/                # 설정 파일
└── data/                  # 데이터셋 준비
```

## 🧪 커스텀 프롬프트 테스트

`simple_demo.py` 수정:

```python
test_prompts = [
    "여기에 원하는 텍스트 입력",
    "또 다른 프롬프트",
    # 더 추가...
]
```

그 다음 실행:
```bash
uv run python simple_demo.py
```

## 🔬 연구 응용 분야

이 구현은 다음 분야에 이상적입니다:
- **효율성 연구**: 뇌 기반 컴퓨팅 연구
- **그린 AI**: 계산 탄소 발자국 감소
- **엣지 배포**: 리소스 제한 장치에서 모델 실행
- **지속 학습**: 망각 없이 학습하는 모델
- **해석 가능성**: 어떤 입력이 깊은 처리가 필요한지 이해

## 📈 향후 방향

1. **생물학적 현실성**: 스파이킹 신경망 구현
2. **하드웨어 최적화**: 뉴로모픽 칩 적응
3. **학습된 라우팅**: 모델이 각 기능을 언제 사용할지 학습
4. **에너지 측정**: 실제 전력 소비 벤치마크
5. **스케일링 연구**: 더 큰 모델(GPT-3 규모)에서 테스트

## 🤝 기여하기

다음 분야의 기여를 환영합니다:
- 새로운 뇌 기반 메커니즘
- 성능 최적화
- 더 나은 평가 지표
- 문서 개선
- 실제 응용 프로그램

## 📚 인용

이 코드를 연구에 사용하는 경우 다음과 같이 인용해주세요:

```bibtex
@software{efficient_gpt,
  title = {Efficient GPT: Brain-Inspired Optimizations for Transformers},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/my-efficient-gpt}
}
```

## 🙏 감사의 말

이 프로젝트는 여러 기초 연구를 바탕으로 합니다:

### 원본 nanoGPT
이 구현은 Andrej Karpathy의 [nanoGPT](https://github.com/karpathy/nanoGPT)를 기반으로 합니다.
```bibtex
@misc{karpathy2022nanogpt,
  author = {Karpathy, Andrej},
  title = {nanoGPT},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  url = {https://github.com/karpathy/nanoGPT}
}
```

### 뇌과학 영감
- 예측 코딩: Rao & Ballard (1999)
- 희소 코딩: Olshausen & Field (1996)  
- 적응적 계산: Graves (2016)
- 지속 학습: Kirkpatrick et al. (2017)

## 📜 라이선스

MIT 라이선스 - 자세한 내용은 [LICENSE](LICENSE) 파일 참조

---

<div align="center">
뇌과학과 AI의 교차점을 탐구하며 만들어졌습니다 🧠
</div>