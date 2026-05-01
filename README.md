# ColBERT Hard Negative Confusion Analysis & LayerRouter Optimization

본 프로젝트는 정보 검색(Information Retrieval) 분야의 SOTA 모델 중 하나인 **ColBERT**에서 빈번하게 발생하는 **Hard Negative (HN) Confusion** 현상을 분석하고, 이를 완화하기 위한 **LayerRouter** 구조를 설계 및 검증하는 연구 코드베이스이다.

## 📌 연구 배경 및 목표

ColBERT와 같은 Late Interaction 모델은 정밀한 토큰 단위 매칭을 수행하지만, 실제 정답(Positive) 문서보다 어휘적 중첩도가 높은 오답(Hard Negative) 문서에 더 높은 점수(MaxSim)를 부여하는 치명적인 실패 모드를 보인다. 

본 연구는 이러한 Confusion 현상이 단순히 최종 출력 단의 문제가 아니라 트랜스포머 인코더 내부 레이어 전반에 걸쳐 누적되는 **표현 표류(Representation Drift)**의 결과라는 가설을 설정하였다. 이를 입증하고 해결하기 위해, 동결된(frozen) ColBERT 인코더의 다층 내부 표현(Multilayer Representations)을 추출하여 Confusion 발생 여부를 사전에 탐지하고 검색 품질을 보정하는 **LayerRouter** 아키텍처를 제안한다.

## 📊 연구 보고서 및 주요 결과

연구 진행에 따라 산출된 공식 리포트 및 분석 결과는 아래 링크에서 확인할 수 있다.

*   👉 **[1차 실험 보고서: LayerRouter 구조 최적화 및 Ablation 연구](./report/ablation_report.md)**
    *   어떤 레이어의 정보를 사용할 것인가? (다층 구조의 필요성)
    *   쿼리와 문서의 표현을 어떻게 융합할 것인가? (Interaction 피처의 압도적 우위)
    *   아키텍처의 복잡도와 정규화 기법의 영향력 분석.
    *   **결과 요약:** Interaction($q \odot d$) 융합 + LayerNorm + 단일 은닉층(256-dim) 구조가 베이스라인 대비 **NDCG 63.0% 향상**이라는 최적의 도메인 일반화 성능(SciFact, NFCorpus, SciDocs)을 달성함을 입증하였다.

## 📁 프로젝트 구조 (Repository Structure)

이 레포지토리는 연구 파이프라인에 따라 논리적으로 분리되어 있다.

```text
nlp_term_project/
├── phase_01/                 # 1차 연구: ColBERT 자가 학습(Introspection) 기반 LayerRouter 최적화
│   ├── shared/                   # 데이터 로더, 모델 래핑(ColBERT Inspector), 공통 평가지표 모듈
│   ├── 01_confusion_analysis/    # Step 1: Baseline Confusion Rate 산출 스크립트
│   ├── 02_layer_signal/          # Step 2: 레이어별 신호 분별력(AUROC) 및 표현 표류 분석
│   ├── 03_router_training/       # Step 3: LayerRouter 최적화 및 MLP 학습 (Ablation Study)
│   ├── 04_intervention/          # Step 4: 학습된 Router를 이용한 실제 검색 결과 랭킹 보정
│   ├── 05_analysis/              # Step 5: 최종 랭킹 결과의 Failure type 심층 정성 분석
│   ├── report/                   # 최종 논문 작성을 위한 실험 결과 보고서 및 Figure 저장소
│   ├── scripts/                  # 결과 분석 및 시각화용 단발성 스크립트 모음
│   └── outputs/                  # 실험 결과(JSON) 및 모델 가중치(Checkpoints) 저장 디렉토리
```

## 🛠 실험 환경 및 데이터셋

*   **데이터셋:** [BEIR Benchmark](https://github.com/beir-cellar/beir) 중 3개 이종 도메인 데이터셋 사용
    *   `SciFact` (과학적 주장 검증)
    *   `NFCorpus` (의학 정보 검색)
    *   `SciDocs` (과학 논문 인용 검색)
*   **검증 프로토콜:** Leave-One-Dataset-Out Cross-Validation (LOOCV) 전략을 사용하여 특정 도메인에 과적합되지 않는 Zero-shot 일반화 성능 측정.
*   **주요 라이브러리:** `torch`, `transformers`, `colbert-ai`, `beir`

## 🚀 향후 과제 (Next Steps)

1.  **Step 4 (Intervention):** Ablation 분석을 통해 확보된 최적의 LayerRouter 아키텍처(`abl2_inter_norm`)를 적용하여 실제 ColBERT 검색 파이프라인에 플러그인(Plug-in) 형태로 결합.
2.  Post-hoc 리랭킹을 수행하여 바닐라 ColBERT 대비 랭킹 지표(NDCG@10, MRR@10)의 최종 향상폭을 검증하고, 논문에 삽입할 최종 Intervention 테이블 생성.
